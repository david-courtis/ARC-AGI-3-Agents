#!/usr/bin/env python3
"""
Dataset Viewer - Step through saved transition datasets interactively.

Usage:
    uv run view_dataset.py <dataset.npz> [--output-dir=<dir>] [--start=<idx>] [--count=<n>]

Commands in interactive mode:
    n/Enter - Next transition
    p       - Previous transition
    g <n>   - Go to transition n
    f       - Filter to show only state-changing transitions
    s       - Show current statistics
    e       - Export current transition as image
    q       - Quit
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

from agents.dataset import TransitionDataset, get_dataset_info
from agents.view_utils import create_grid_image, create_transition_image


def print_transition(dataset: TransitionDataset, idx: int) -> None:
    """Print details of a single transition."""
    item = dataset[idx]

    # Get raw data
    move_type = str(dataset.move_types[idx])
    move = int(dataset.moves[idx])
    changed = bool(dataset.changed[idx])
    solution = bool(dataset.solutions[idx])
    score_before = int(dataset.scores_before[idx])
    score_after = int(dataset.scores_after[idx])
    state_hash = str(dataset.hashes[idx])[:8]
    move_num = int(dataset.move_nums[idx])

    # Get move sequence if available
    if dataset.move_sequences is not None:
        seq = dataset.move_sequences[idx]
        seq_str = " -> ".join(str(a) for a in seq) if len(seq) > 0 else "(start)"
    else:
        seq_str = "(not available)"

    print(f"\n{'='*60}")
    print(f"Transition {idx + 1}/{len(dataset)}")
    print(f"{'='*60}")
    print(f"Move #:        {move_num}")
    print(f"Action:        {move_type} (value: {move})")
    print(f"Prev sequence: {seq_str}")
    print(f"State changed: {'YES' if changed else 'NO'}")
    print(f"Is solution:   {'YES - WIN!' if solution else 'No'}")
    print(f"Score:         {score_before} -> {score_after} (delta: {score_after - score_before})")
    print(f"State hash:    {state_hash}...")

    # Show a text representation of the grids
    before = dataset.before_states[idx]
    after = dataset.after_states[idx]

    # Show difference summary
    if changed:
        diff = before != after
        n_changed = diff.sum()
        print(f"Pixels changed: {n_changed} / {64*64} ({100*n_changed/(64*64):.1f}%)")
    print(f"{'='*60}")


def export_transition(dataset: TransitionDataset, idx: int, output_dir: str) -> str:
    """Export a transition as an image."""
    before = dataset.before_states[idx]
    after = dataset.after_states[idx]
    move_type = str(dataset.move_types[idx])
    changed = "CHANGED" if dataset.changed[idx] else "NO_CHANGE"
    solution = " - WIN" if dataset.solutions[idx] else ""

    action_info = f"#{idx+1}: {move_type} | {changed}{solution}"

    img = create_transition_image(before, after, action_info, cell_size=8)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"transition_{idx:05d}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)

    return filepath


def interactive_mode(dataset: TransitionDataset, output_dir: str, start_idx: int = 0) -> None:
    """Run interactive stepping mode."""
    idx = start_idx
    filter_changed = False
    indices = list(range(len(dataset)))

    def get_filtered_indices():
        if filter_changed:
            return [i for i in range(len(dataset)) if dataset.changed[i]]
        return list(range(len(dataset)))

    indices = get_filtered_indices()
    if not indices:
        print("No transitions match the current filter.")
        return

    # Find starting position in filtered list
    if idx in indices:
        pos = indices.index(idx)
    else:
        pos = 0
        idx = indices[0]

    print("\nInteractive Dataset Viewer")
    print("Commands: [n]ext, [p]rev, [g]oto <n>, [f]ilter, [s]tats, [e]xport, [q]uit")

    while True:
        print_transition(dataset, idx)

        try:
            cmd = input("\n> ").strip().lower()
        except EOFError:
            break

        if cmd in ('', 'n', 'next'):
            pos = min(pos + 1, len(indices) - 1)
            idx = indices[pos]
        elif cmd in ('p', 'prev'):
            pos = max(pos - 1, 0)
            idx = indices[pos]
        elif cmd.startswith('g ') or cmd.startswith('goto '):
            try:
                target = int(cmd.split()[1]) - 1  # 1-indexed input
                if 0 <= target < len(dataset):
                    idx = target
                    if idx in indices:
                        pos = indices.index(idx)
                    else:
                        print(f"Transition {target+1} doesn't match current filter, showing anyway.")
                else:
                    print(f"Invalid index. Range: 1-{len(dataset)}")
            except (ValueError, IndexError):
                print("Usage: g <transition_number>")
        elif cmd in ('f', 'filter'):
            filter_changed = not filter_changed
            indices = get_filtered_indices()
            if not indices:
                print("No transitions match filter. Disabling filter.")
                filter_changed = False
                indices = get_filtered_indices()
            pos = 0
            idx = indices[0]
            status = "ON (showing only state-changing)" if filter_changed else "OFF (showing all)"
            print(f"Filter: {status} - {len(indices)} transitions")
        elif cmd in ('s', 'stats'):
            n_total = len(dataset)
            n_changed = sum(1 for i in range(n_total) if dataset.changed[i])
            n_solutions = sum(1 for i in range(n_total) if dataset.solutions[i])
            print(f"\nDataset Statistics:")
            print(f"  Total transitions: {n_total}")
            print(f"  State-changing:    {n_changed} ({100*n_changed/n_total:.1f}%)")
            print(f"  Solutions (WIN):   {n_solutions}")
            print(f"  Current filter:    {'Changed only' if filter_changed else 'All'}")
            print(f"  Filtered count:    {len(indices)}")
        elif cmd in ('e', 'export'):
            filepath = export_transition(dataset, idx, output_dir)
            print(f"Exported to: {filepath}")
        elif cmd in ('q', 'quit', 'exit'):
            break
        else:
            print("Unknown command. Use: [n]ext, [p]rev, [g]oto <n>, [f]ilter, [s]tats, [e]xport, [q]uit")


def batch_export(dataset: TransitionDataset, output_dir: str, count: Optional[int] = None) -> None:
    """Export all (or first N) transitions as images."""
    n = min(count, len(dataset)) if count else len(dataset)

    print(f"Exporting {n} transitions to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n):
        filepath = export_transition(dataset, i, output_dir)
        if (i + 1) % 100 == 0:
            print(f"  Exported {i + 1}/{n}...")

    print(f"Done! Exported {n} transitions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="View and step through saved transition datasets"
    )
    parser.add_argument(
        "dataset",
        help="Path to .npz dataset file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="dataset_viz",
        help="Directory for exported images (default: dataset_viz)"
    )
    parser.add_argument(
        "-s", "--start",
        type=int,
        default=0,
        help="Starting transition index (default: 0)"
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=None,
        help="Number of transitions to export in batch mode"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch export mode (export all as images, no interaction)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Just print dataset info and exit"
    )
    parser.add_argument(
        "--include-sequences",
        action="store_true",
        help="Load move sequences (slower but more info)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Just print info
    if args.info:
        info = get_dataset_info(args.dataset)
        print(f"\nDataset: {info['path']}")
        print(f"Transitions: {info['n_transitions']}")
        print(f"State-changing: {info['n_changed']}")
        print(f"Solutions: {info['n_solutions']}")
        print(f"Games: {info['unique_games']}")
        print(f"Action distribution: {info['action_distribution']}")
        if 'generation_stats' in info:
            print(f"Generation stats: {info['generation_stats']}")
        return

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = TransitionDataset(
        args.dataset,
        one_hot_encode=False,  # Keep raw values for visualization
        include_sequences=args.include_sequences,
    )
    print(f"Loaded {len(dataset)} transitions")

    if args.batch:
        batch_export(dataset, args.output_dir, args.count)
    else:
        interactive_mode(dataset, args.output_dir, args.start)


if __name__ == "__main__":
    main()
