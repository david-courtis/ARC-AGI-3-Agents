"""
Epistemic state: continuous confidence tracking for object–action interactions.

Replaces the binary KT/KF/UK model with a richer representation:

1. Each (sprite_type, action) cell stores ALL observations, not just a summary.
2. Confidence is derived from context diversity × consistency, not a threshold count.
3. Observations carry full context snapshots (what was nearby, what was selected).
4. Inconsistencies are preserved, not collapsed — they signal conditional effects.

Key insight: "Known True" is a fiction. We never exhaustively test all states.
Instead we track:
  - How many UNIQUE CONTEXTS have we observed this (type, action) in?
  - How CONSISTENT is the effect across those contexts?
  - When inconsistency occurs, what was DIFFERENT about the context?

The exploration layer uses this to prefer actions with low context diversity
(we haven't seen enough variety) or high inconsistency (there's a conditional
effect we need to disambiguate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Context: what the world looked like when an observation was made
# =============================================================================


@dataclass(frozen=True)
class ObservationContext:
    """
    A fingerprint of the game state when a transition was observed.

    Two observations have different contexts if the spatial arrangement
    of objects around the target sprite was different. This is how we
    measure "have we tested this action in diverse situations?"
    """
    # Types of objects within interaction range (sorted tuple for hashability)
    nearby_type_ids: tuple[int, ...]
    # Relative positions of nearby objects (sorted for consistency)
    nearby_relative_positions: tuple[tuple[int, int], ...]
    # Was something selected?
    selected_type_id: int | None = None
    # Player position relative to the target (if target is not the player)
    player_relative: tuple[int, int] | None = None

    @staticmethod
    def from_world_snapshot(
        target_position: tuple[int, int],
        nearby_sprites: list[dict],
        selected_type_id: int | None = None,
        player_position: tuple[int, int] | None = None,
    ) -> ObservationContext:
        type_ids = tuple(sorted(s["type_id"] for s in nearby_sprites))
        rel_positions = tuple(sorted(
            (s["position"][0] - target_position[0],
             s["position"][1] - target_position[1])
            for s in nearby_sprites
        ))
        player_rel = None
        if player_position is not None:
            player_rel = (
                player_position[0] - target_position[0],
                player_position[1] - target_position[1],
            )
        return ObservationContext(
            nearby_type_ids=type_ids,
            nearby_relative_positions=rel_positions,
            selected_type_id=selected_type_id,
            player_relative=player_rel,
        )


# =============================================================================
# TransitionRecord: a single observation with full context
# =============================================================================


@dataclass
class TransitionRecord:
    """
    A single observed transition for a (sprite_type, action) pair.

    Stores both the effect AND the context, so we can analyze
    "under what conditions does this effect occur?"
    """
    frame_index: int
    action_id: int
    sprite_type_id: int
    track_id: int

    # Before/after state
    before_position: tuple[int, int]
    after_position: tuple[int, int]
    before_properties: dict[str, Any] = field(default_factory=dict)
    after_properties: dict[str, Any] = field(default_factory=dict)

    # Computed effect
    displacement: tuple[int, int] = (0, 0)
    had_effect: bool = False

    # Full context snapshot
    context: ObservationContext | None = None
    nearby_types: list[int] = field(default_factory=list)
    selected_type_id: int | None = None

    @property
    def is_movement(self) -> bool:
        return self.displacement != (0, 0)

    @property
    def effect_signature(self) -> str:
        """
        A hashable string summarizing WHAT happened (not WHERE or WHEN).
        Used to group records by observed effect.
        """
        if not self.had_effect:
            return "no_effect"
        if self.is_movement:
            return f"move({self.displacement[0]:+d},{self.displacement[1]:+d})"
        return "state_change"


# =============================================================================
# TransitionEffect: accumulated knowledge for one (type, action) cell
# =============================================================================


@dataclass
class TransitionEffect:
    """
    Everything we know about a (sprite_type, action) pair.

    Instead of collapsing to KT/KF/UK, we keep all observations and derive
    confidence metrics on demand. The exploration layer reads these metrics
    to decide where to invest more testing.
    """
    sprite_type_id: int
    action_id: int

    # All raw observations (never discarded)
    records: list[TransitionRecord] = field(default_factory=list)

    @property
    def observation_count(self) -> int:
        return len(self.records)

    @property
    def unique_contexts(self) -> int:
        """Number of distinct spatial contexts this was observed in."""
        seen: set[ObservationContext] = set()
        for r in self.records:
            if r.context is not None:
                seen.add(r.context)
        return max(len(seen), 1) if self.records else 0

    @property
    def effect_signatures(self) -> dict[str, int]:
        """
        Count of each observed effect type.
        e.g., {"move(+0,+3)": 5, "no_effect": 2} means it moved 5 times
        and was blocked 2 times.
        """
        counts: dict[str, int] = {}
        for r in self.records:
            sig = r.effect_signature
            counts[sig] = counts.get(sig, 0) + 1
        return counts

    @property
    def dominant_effect(self) -> str:
        """The most frequently observed effect signature."""
        sigs = self.effect_signatures
        if not sigs:
            return "unobserved"
        return max(sigs, key=sigs.get)

    @property
    def is_conditional(self) -> bool:
        """True if we've observed different effects in different contexts."""
        return len(self.effect_signatures) > 1

    @property
    def consistency(self) -> float:
        """
        Fraction of observations matching the dominant effect.
        1.0 = perfectly consistent. 0.5 = two equally common outcomes.
        """
        sigs = self.effect_signatures
        if not sigs:
            return 0.0
        total = sum(sigs.values())
        return max(sigs.values()) / total

    @property
    def context_diversity(self) -> float:
        """
        Ratio of unique contexts to total observations.
        High = we've tested in many different situations.
        Low = we keep testing the same situation.
        """
        if self.observation_count == 0:
            return 0.0
        return self.unique_contexts / self.observation_count

    @property
    def confidence(self) -> float:
        """
        Overall confidence in our understanding of this effect.

        Combines consistency (are the results stable?) with diversity
        (have we tested in varied situations?).

        High confidence = consistent across diverse contexts.
        Low confidence = few observations, or inconsistent, or low diversity.
        """
        if self.observation_count == 0:
            return 0.0

        # Base: how many observations do we have? Diminishing returns.
        obs_factor = min(1.0, self.observation_count / 5.0)

        # Consistency: do we always see the same effect?
        cons = self.consistency

        # Diversity: have we tested in different contexts?
        div = min(1.0, self.unique_contexts / 3.0)

        return obs_factor * cons * (0.5 + 0.5 * div)

    @property
    def exploration_priority(self) -> float:
        """
        How much should the explorer prioritize testing this cell?
        Higher = we know less, should test more.

        Priority is HIGH when:
        - Few observations (we've barely tested)
        - Low context diversity (always tested same situation)

        Priority DECAYS with observation count — even conditional effects
        become low-priority once we've seen enough cases. The conditionality
        itself is useful information for synthesis; we don't need to keep
        testing forever.
        """
        if self.observation_count == 0:
            return 1.0  # untested → highest priority

        # Diminishing returns: more observations → lower priority
        # Drops from 1.0 to ~0.1 over 10 observations
        obs_decay = 1.0 / (1.0 + self.observation_count * 0.3)

        # Low diversity boost: if all observations are from the same context,
        # we should try from a different position
        diversity_boost = 0.0
        if self.unique_contexts < 3 and self.observation_count < 8:
            diversity_boost = 0.3

        return min(1.0, obs_decay + diversity_boost)

    def record_observation(self, record: TransitionRecord) -> None:
        """Add an observation. No thresholds, no state transitions — just accumulate."""
        self.records.append(record)

    def describe(self) -> str:
        """Human-readable summary."""
        if self.observation_count == 0:
            return "unobserved"

        sigs = self.effect_signatures
        parts = []
        for sig, count in sorted(sigs.items(), key=lambda x: -x[1]):
            parts.append(f"{sig}×{count}")

        cond = " [CONDITIONAL]" if self.is_conditional else ""
        return (
            f"{' | '.join(parts)}{cond} "
            f"(conf={self.confidence:.0%}, "
            f"ctx={self.unique_contexts}, "
            f"obs={self.observation_count})"
        )

    def describe_short(self) -> str:
        """Compact display for the epistemic matrix."""
        if self.observation_count == 0:
            return "  --    "
        dom = self.dominant_effect
        n = self.observation_count
        c = self.confidence
        if dom == "no_effect":
            return f"  nop/{n} "
        elif dom.startswith("move"):
            disp = dom[4:]  # "(+0,+3)"
            return f"  {disp}/{n}"
        return f"  ?/{n}  "


# =============================================================================
# Full epistemic state
# =============================================================================


class EpistemicState:
    """
    Complete epistemic model with continuous confidence, not binary KT/KF/UK.

    The exploration layer queries exploration_priority to decide what to test.
    The synthesis layer queries dominant effects + context for code generation.
    """

    def __init__(self):
        self._effects: dict[tuple[int, int], TransitionEffect] = {}
        self._properties: dict[tuple[int, str], Any] = {}

    @property
    def effects(self) -> dict[tuple[int, int], TransitionEffect]:
        return dict(self._effects)

    def initialize_for_types(
        self,
        type_ids: list[int],
        action_ids: list[int],
    ) -> None:
        """Create empty cells for all (type, action) combinations."""
        for tid in type_ids:
            for aid in action_ids:
                key = (tid, aid)
                if key not in self._effects:
                    self._effects[key] = TransitionEffect(
                        sprite_type_id=tid, action_id=aid,
                    )

    def record_transition(self, record: TransitionRecord) -> TransitionEffect:
        """Record an observation."""
        key = (record.sprite_type_id, record.action_id)
        if key not in self._effects:
            self._effects[key] = TransitionEffect(
                sprite_type_id=record.sprite_type_id,
                action_id=record.action_id,
            )
        eff = self._effects[key]
        eff.record_observation(record)
        return eff

    def get_effect(self, type_id: int, action_id: int) -> TransitionEffect | None:
        return self._effects.get((type_id, action_id))

    def set_property(self, type_id: int, prop: str, value: Any) -> None:
        self._properties[(type_id, prop)] = value

    def get_property(self, type_id: int, prop: str) -> Any:
        return self._properties.get((type_id, prop))

    # =========================================================================
    # Exploration queries
    # =========================================================================

    def get_exploration_targets(self) -> list[tuple[int, int, float]]:
        """
        All (type_id, action_id, priority) sorted by exploration priority.
        Highest priority first = least understood.
        """
        targets = [
            (eff.sprite_type_id, eff.action_id, eff.exploration_priority)
            for eff in self._effects.values()
        ]
        targets.sort(key=lambda x: -x[2])
        return targets

    def get_high_confidence_effects(self, threshold: float = 0.6) -> list[TransitionEffect]:
        """Effects with confidence above threshold."""
        return [
            e for e in self._effects.values()
            if e.confidence >= threshold and e.observation_count > 0
        ]

    def get_conditional_effects(self) -> list[TransitionEffect]:
        """Effects that show different outcomes in different contexts."""
        return [e for e in self._effects.values() if e.is_conditional]

    def coverage(self) -> tuple[int, int, int]:
        """Returns (high_conf, low_conf, unobserved) counts."""
        high = sum(1 for e in self._effects.values() if e.confidence >= 0.6)
        low = sum(1 for e in self._effects.values()
                  if 0 < e.observation_count and e.confidence < 0.6)
        unobs = sum(1 for e in self._effects.values() if e.observation_count == 0)
        return (high, low, unobs)

    def coverage_ratio(self) -> float:
        """Fraction of cells that have been observed at all."""
        total = len(self._effects)
        observed = sum(1 for e in self._effects.values() if e.observation_count > 0)
        return observed / total if total > 0 else 0.0

    def mean_confidence(self) -> float:
        """Average confidence across all cells."""
        if not self._effects:
            return 0.0
        return sum(e.confidence for e in self._effects.values()) / len(self._effects)

    # =========================================================================
    # Display
    # =========================================================================

    def describe(self, type_names: dict[int, str] | None = None) -> str:
        """Human-readable epistemic matrix with confidence values."""
        if not self._effects:
            return "EpistemicState: empty"

        type_ids = sorted({e.sprite_type_id for e in self._effects.values()})
        action_ids = sorted({e.action_id for e in self._effects.values()})
        names = type_names or {}

        lines = ["EpistemicState:"]

        # Header
        header = f"  {'type':<12}" + "".join(f"  ACT{a:<8}" for a in action_ids)
        lines.append(header)

        for tid in type_ids:
            tname = names.get(tid, f"type_{tid}")
            row = [f"  {tname:<12}"]
            for aid in action_ids:
                eff = self._effects.get((tid, aid))
                if eff is None:
                    row.append("  --      ")
                else:
                    row.append(eff.describe_short())
            lines.append("".join(row))

        hi, lo, un = self.coverage()
        lines.append(
            f"  Observed: {hi + lo}/{hi + lo + un} cells, "
            f"mean confidence: {self.mean_confidence():.0%}"
        )

        # Show conditional effects
        conds = self.get_conditional_effects()
        if conds:
            lines.append(f"  Conditional effects ({len(conds)}):")
            for e in conds:
                tname = names.get(e.sprite_type_id, f"type_{e.sprite_type_id}")
                lines.append(f"    ACTION{e.action_id} on {tname}: {e.describe()}")

        return "\n".join(lines)
