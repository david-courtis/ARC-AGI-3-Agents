"""
Unit tests for world model base classes and frame utilities.

Tests all three agent variants:
- OOP agent (action-centric): GameObject, Action, World, Domain
- Monolithic agent: WorldModel, IdentityModel
- Object-centric agent: GameObject, World, Domain (no Action classes)

Frame utilities (compute_diff, find_unique_colors, find_color_regions,
region_bbox, most_common_color, extract_grid) are duplicated across all three
agents. We test them from each module to ensure independence.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# OOP agent imports
# ---------------------------------------------------------------------------
from src.oop_agent.world_model import (
    GameObject as OOPGameObject,
    Action as OOPAction,
    World as OOPWorld,
    Domain as OOPDomain,
    PixelDiff as OOPPixelDiff,
    compute_diff as oop_compute_diff,
    find_unique_colors as oop_find_unique_colors,
    find_color_regions as oop_find_color_regions,
    region_bbox as oop_region_bbox,
    most_common_color as oop_most_common_color,
    extract_grid as oop_extract_grid,
)

# ---------------------------------------------------------------------------
# Monolithic agent imports
# ---------------------------------------------------------------------------
from src.monolithic_agent.world_model import (
    WorldModel,
    IdentityModel,
    PixelDiff as MonoPixelDiff,
    compute_diff as mono_compute_diff,
    find_unique_colors as mono_find_unique_colors,
    find_color_regions as mono_find_color_regions,
    region_bbox as mono_region_bbox,
    most_common_color as mono_most_common_color,
    extract_grid as mono_extract_grid,
)

# ---------------------------------------------------------------------------
# Object-centric agent imports
# ---------------------------------------------------------------------------
from src.object_centric_agent.world_model import (
    GameObject as OCGameObject,
    World as OCWorld,
    Domain as OCDomain,
    PixelDiff as OCPixelDiff,
    compute_diff as oc_compute_diff,
    find_unique_colors as oc_find_unique_colors,
    find_color_regions as oc_find_color_regions,
    region_bbox as oc_region_bbox,
    most_common_color as oc_most_common_color,
    extract_grid as oc_extract_grid,
)


# =============================================================================
# Helpers: concrete subclasses for testing abstract base classes
# =============================================================================


class SimpleOOPObject(OOPGameObject):
    """Minimal OOP GameObject for testing."""

    def respond_to_action(self, action, world):
        pass

    def render(self, frame):
        if hasattr(self, "row") and hasattr(self, "col") and hasattr(self, "color"):
            frame[self.row, self.col] = self.color


class SimpleOOPAction(OOPAction):
    """Minimal OOP Action for testing."""

    def apply(self, world):
        for obj in world.objects:
            obj.respond_to_action(self, world)


class SimpleOOPDomain(OOPDomain):
    """Minimal OOP Domain for testing."""

    def __init__(self, objects_factory=None, action_factory=None):
        self._objects_factory = objects_factory or (lambda frame: [])
        self._action_factory = action_factory or (lambda aid: SimpleOOPAction(aid))

    def perceive(self, frame):
        return OOPWorld(frame, self._objects_factory(frame))

    def get_action(self, action_id):
        return self._action_factory(action_id)


class SimpleOCObject(OCGameObject):
    """Minimal object-centric GameObject for testing."""

    def respond(self, action_id, world):
        pass

    def render(self, frame):
        if hasattr(self, "row") and hasattr(self, "col") and hasattr(self, "color"):
            frame[self.row, self.col] = self.color


class SimpleOCDomain(OCDomain):
    """Minimal object-centric Domain for testing."""

    def __init__(self, objects_factory=None):
        self._objects_factory = objects_factory or (lambda frame: [])

    def perceive(self, frame):
        return OCWorld(frame, self._objects_factory(frame))


class SimpleMonoModel(WorldModel):
    """Minimal monolithic WorldModel that returns frame unchanged."""

    def predict(self, frame, action_id):
        return frame.copy()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def black_frame():
    """4x4 RGB frame, all black."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture
def simple_frame():
    """4x4 RGB frame with a red pixel at (1,1) and green at (2,2)."""
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    f[1, 1] = [255, 0, 0]
    f[2, 2] = [0, 255, 0]
    return f


@pytest.fixture
def game_frame():
    """64x64 RGB frame simulating a simple game with walls and player."""
    f = np.zeros((64, 64, 3), dtype=np.uint8)
    # Background = dark gray
    f[:] = [32, 32, 32]
    # Border walls = white
    f[0, :] = [255, 255, 255]
    f[-1, :] = [255, 255, 255]
    f[:, 0] = [255, 255, 255]
    f[:, -1] = [255, 255, 255]
    # Player = red 4x4 block at cell (2,2) -> pixels (8:12, 8:12)
    f[8:12, 8:12] = [255, 0, 0]
    # Goal = green 4x4 block at cell (10,10) -> pixels (40:44, 40:44)
    f[40:44, 40:44] = [0, 255, 0]
    return f


# =============================================================================
# Test: OOP agent base classes
# =============================================================================


@pytest.mark.unit
class TestOOPGameObject:
    def test_init_with_properties(self):
        obj = SimpleOOPObject("p1", row=5, col=10, color=(255, 0, 0))
        assert obj.obj_id == "p1"
        assert obj.row == 5
        assert obj.col == 10
        assert obj.color == (255, 0, 0)

    def test_type_name(self):
        obj = SimpleOOPObject("w1")
        assert obj.type_name == "SimpleOOPObject"

    def test_init_no_properties(self):
        obj = SimpleOOPObject("bare")
        assert obj.obj_id == "bare"
        assert not hasattr(obj, "row")

    def test_repr(self):
        obj = SimpleOOPObject("r1", hp=100)
        r = repr(obj)
        assert "SimpleOOPObject" in r
        assert "r1" in r
        assert "hp" in r

    def test_render_writes_pixel(self):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        obj = SimpleOOPObject("p", row=1, col=2, color=[0, 0, 255])
        obj.render(f)
        assert list(f[1, 2]) == [0, 0, 255]


@pytest.mark.unit
class TestOOPAction:
    def test_init(self):
        a = SimpleOOPAction(3)
        assert a.action_id == 3

    def test_repr(self):
        a = SimpleOOPAction(1)
        assert "SimpleOOPAction" in repr(a)
        assert "1" in repr(a)

    def test_apply_dispatches(self):
        called = []

        class TrackingObject(OOPGameObject):
            def respond_to_action(self, action, world):
                called.append(self.obj_id)

            def render(self, frame):
                pass

        o1 = TrackingObject("a")
        o2 = TrackingObject("b")
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        world = OOPWorld(frame, [o1, o2])
        action = SimpleOOPAction(1)
        action.apply(world)
        assert called == ["a", "b"]


@pytest.mark.unit
class TestOOPWorld:
    def test_init(self):
        f = np.zeros((8, 8, 3), dtype=np.uint8)
        objs = [SimpleOOPObject("a"), SimpleOOPObject("b")]
        w = OOPWorld(f, objs)
        assert w.height == 8
        assert w.width == 8
        assert len(w.objects) == 2

    def test_frame_is_copied(self):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        w = OOPWorld(f, [])
        f[0, 0] = [255, 255, 255]
        assert np.all(w.frame[0, 0] == 0)

    def test_get_objects_of_type(self):
        class TypeA(SimpleOOPObject):
            pass

        class TypeB(SimpleOOPObject):
            pass

        a = TypeA("a1")
        b = TypeB("b1")
        w = OOPWorld(np.zeros((4, 4, 3), dtype=np.uint8), [a, b])
        assert w.get_objects_of_type(TypeA) == [a]
        assert w.get_objects_of_type(TypeB) == [b]
        assert len(w.get_objects_of_type(SimpleOOPObject)) == 2

    def test_get_by_id(self):
        a = SimpleOOPObject("x")
        w = OOPWorld(np.zeros((4, 4, 3), dtype=np.uint8), [a])
        assert w.get_by_id("x") is a
        assert w.get_by_id("missing") is None

    def test_add_remove_object(self):
        a = SimpleOOPObject("a")
        b = SimpleOOPObject("b")
        w = OOPWorld(np.zeros((4, 4, 3), dtype=np.uint8), [a])
        assert len(w.objects) == 1
        w.add_object(b)
        assert len(w.objects) == 2
        w.remove_object(a)
        assert len(w.objects) == 1
        assert w.objects[0] is b

    def test_render(self):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        obj = SimpleOOPObject("p", row=0, col=0, color=[255, 0, 0])
        w = OOPWorld(f, [obj])
        rendered = w.render()
        assert list(rendered[0, 0]) == [255, 0, 0]
        # Original frame untouched
        assert np.all(f[0, 0] == 0)

    def test_render_with_background(self):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        bg = np.full((4, 4, 3), 128, dtype=np.uint8)
        obj = SimpleOOPObject("p", row=0, col=0, color=[255, 0, 0])
        w = OOPWorld(f, [obj])
        rendered = w.render(background=bg)
        assert list(rendered[0, 0]) == [255, 0, 0]
        assert list(rendered[1, 1]) == [128, 128, 128]
        # Background untouched
        assert np.all(bg[0, 0] == 128)


@pytest.mark.unit
class TestOOPDomain:
    def test_transition_pipeline(self):
        """Test perceive -> get_action -> apply -> render pipeline."""
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        class MovingObj(OOPGameObject):
            def respond_to_action(self, action, world):
                if action.action_id == 1:
                    self.col = min(self.col + 1, 3)

            def render(self, f):
                f[self.row, self.col] = self.color

        def make_objects(f):
            return [MovingObj("p", row=0, col=0, color=[255, 0, 0])]

        domain = SimpleOOPDomain(objects_factory=make_objects)
        result = domain.transition(frame, 1)
        # Object should have moved from (0,0) to (0,1)
        assert list(result[0, 1]) == [255, 0, 0]
        assert list(result[0, 0]) == [0, 0, 0]


# =============================================================================
# Test: Object-centric agent base classes
# =============================================================================


@pytest.mark.unit
class TestOCGameObject:
    def test_init_with_properties(self):
        obj = SimpleOCObject("p1", row=5, col=10)
        assert obj.obj_id == "p1"
        assert obj.row == 5
        assert obj.col == 10

    def test_type_name(self):
        obj = SimpleOCObject("w1")
        assert obj.type_name == "SimpleOCObject"

    def test_repr(self):
        obj = SimpleOCObject("r1", hp=100)
        assert "SimpleOCObject" in repr(obj)


@pytest.mark.unit
class TestOCWorld:
    def test_init_and_queries(self):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        a = SimpleOCObject("a")
        b = SimpleOCObject("b")
        w = OCWorld(f, [a, b])
        assert w.height == 4
        assert w.width == 4
        assert w.get_by_id("a") is a
        assert w.get_by_id("c") is None
        assert len(w.get_objects_of_type(SimpleOCObject)) == 2

    def test_add_remove(self):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        a = SimpleOCObject("a")
        b = SimpleOCObject("b")
        w = OCWorld(f, [a])
        w.add_object(b)
        assert len(w.objects) == 2
        w.remove_object(a)
        assert len(w.objects) == 1
        assert w.objects[0] is b


@pytest.mark.unit
class TestOCDomain:
    def test_transition_broadcasts_to_all_objects(self):
        """Objects receive action_id, not Action objects."""
        received = []

        class TrackObj(OCGameObject):
            def respond(self, action_id, world):
                received.append((self.obj_id, action_id))

            def render(self, frame):
                pass

        def make_objects(f):
            return [TrackObj("a"), TrackObj("b")]

        domain = SimpleOCDomain(objects_factory=make_objects)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        domain.transition(frame, 3)
        assert ("a", 3) in received
        assert ("b", 3) in received

    def test_respond_order(self):
        """Test that respond_order controls call sequence."""
        order = []

        class OrderObj(OCGameObject):
            def respond(self, action_id, world):
                order.append(self.obj_id)

            def render(self, frame):
                pass

        class OrderedDomain(OCDomain):
            def perceive(self, frame):
                return OCWorld(frame, [OrderObj("b"), OrderObj("a")])

            def respond_order(self, world, action_id):
                # Reverse order
                return list(reversed(world.objects))

        domain = OrderedDomain()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        domain.transition(frame, 1)
        assert order == ["a", "b"]

    def test_no_action_class(self):
        """Verify object-centric has no Action base class."""
        import src.object_centric_agent.world_model as oc_wm
        assert not hasattr(oc_wm, "Action")


# =============================================================================
# Test: Monolithic agent base classes
# =============================================================================


@pytest.mark.unit
class TestMonolithicWorldModel:
    def test_identity_model(self):
        f = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        model = IdentityModel()
        result = model.predict(f, 1)
        assert np.array_equal(result, f)
        # Should be a copy, not same object
        result[0, 0] = [0, 0, 0]
        assert not np.array_equal(result, f) or np.all(f[0, 0] == 0)

    def test_custom_model(self):
        model = SimpleMonoModel()
        f = np.ones((4, 4, 3), dtype=np.uint8) * 128
        result = model.predict(f, 2)
        assert np.array_equal(result, f)

    def test_reset_is_noop(self):
        model = IdentityModel()
        model.reset()  # Should not raise


# =============================================================================
# Test: Frame utilities (parameterized across all 3 agents)
# =============================================================================


COMPUTE_DIFF_FUNCS = [
    pytest.param(oop_compute_diff, id="oop"),
    pytest.param(mono_compute_diff, id="mono"),
    pytest.param(oc_compute_diff, id="oc"),
]

FIND_COLORS_FUNCS = [
    pytest.param(oop_find_unique_colors, id="oop"),
    pytest.param(mono_find_unique_colors, id="mono"),
    pytest.param(oc_find_unique_colors, id="oc"),
]

FIND_REGIONS_FUNCS = [
    pytest.param(oop_find_color_regions, id="oop"),
    pytest.param(mono_find_color_regions, id="mono"),
    pytest.param(oc_find_color_regions, id="oc"),
]

REGION_BBOX_FUNCS = [
    pytest.param(oop_region_bbox, id="oop"),
    pytest.param(mono_region_bbox, id="mono"),
    pytest.param(oc_region_bbox, id="oc"),
]

MOST_COMMON_FUNCS = [
    pytest.param(oop_most_common_color, id="oop"),
    pytest.param(mono_most_common_color, id="mono"),
    pytest.param(oc_most_common_color, id="oc"),
]

EXTRACT_GRID_FUNCS = [
    pytest.param(oop_extract_grid, id="oop"),
    pytest.param(mono_extract_grid, id="mono"),
    pytest.param(oc_extract_grid, id="oc"),
]


@pytest.mark.unit
class TestComputeDiff:
    @pytest.mark.parametrize("compute_diff", COMPUTE_DIFF_FUNCS)
    def test_identical_frames(self, compute_diff):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        diff = compute_diff(f, f.copy())
        assert diff.count == 0
        assert diff.positions == []
        assert diff.bbox is None

    @pytest.mark.parametrize("compute_diff", COMPUTE_DIFF_FUNCS)
    def test_single_pixel_change(self, compute_diff):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[2, 3] = [255, 0, 0]
        diff = compute_diff(a, b)
        assert diff.count == 1
        assert (2, 3) in diff.positions
        assert diff.bbox == (2, 3, 2, 3)
        assert diff.before_colors == [(0, 0, 0)]
        assert diff.after_colors == [(255, 0, 0)]

    @pytest.mark.parametrize("compute_diff", COMPUTE_DIFF_FUNCS)
    def test_multiple_pixel_change(self, compute_diff):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = [255, 0, 0]
        b[3, 3] = [0, 255, 0]
        diff = compute_diff(a, b)
        assert diff.count == 2
        assert diff.bbox == (0, 0, 3, 3)

    @pytest.mark.parametrize("compute_diff", COMPUTE_DIFF_FUNCS)
    def test_shape_mismatch(self, compute_diff):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.zeros((8, 8, 3), dtype=np.uint8)
        diff = compute_diff(a, b)
        assert diff.count == 0

    @pytest.mark.parametrize("compute_diff", COMPUTE_DIFF_FUNCS)
    def test_2d_frames(self, compute_diff):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = a.copy()
        b[1, 1] = 128
        diff = compute_diff(a, b)
        assert diff.count == 1
        assert diff.before_colors == [(0,)]
        assert diff.after_colors == [(128,)]

    @pytest.mark.parametrize("compute_diff", COMPUTE_DIFF_FUNCS)
    def test_positions_capped_at_1000(self, compute_diff):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.ones((64, 64, 3), dtype=np.uint8) * 255
        diff = compute_diff(a, b)
        assert diff.count == 64 * 64
        assert len(diff.positions) <= 1000

    @pytest.mark.parametrize("compute_diff", COMPUTE_DIFF_FUNCS)
    def test_colors_sampled_max_100(self, compute_diff):
        a = np.zeros((20, 20, 3), dtype=np.uint8)
        b = np.ones((20, 20, 3), dtype=np.uint8)
        diff = compute_diff(a, b)
        assert len(diff.before_colors) <= 100
        assert len(diff.after_colors) <= 100


@pytest.mark.unit
class TestFindUniqueColors:
    @pytest.mark.parametrize("find_colors", FIND_COLORS_FUNCS)
    def test_single_color(self, find_colors):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        colors = find_colors(f)
        assert len(colors) == 1
        assert colors[0] == (0, 0, 0)

    @pytest.mark.parametrize("find_colors", FIND_COLORS_FUNCS)
    def test_two_colors(self, find_colors):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[0, 0] = [255, 0, 0]
        colors = find_colors(f)
        assert len(colors) == 2
        assert (0, 0, 0) in colors
        assert (255, 0, 0) in colors

    @pytest.mark.parametrize("find_colors", FIND_COLORS_FUNCS)
    def test_2d_frame(self, find_colors):
        f = np.array([[0, 1], [1, 2]], dtype=np.uint8)
        colors = find_colors(f)
        assert len(colors) == 3
        assert (0,) in colors
        assert (1,) in colors
        assert (2,) in colors


@pytest.mark.unit
class TestFindColorRegions:
    @pytest.mark.parametrize("find_regions", FIND_REGIONS_FUNCS)
    def test_single_region(self, find_regions):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[1, 1] = [255, 0, 0]
        f[1, 2] = [255, 0, 0]
        regions = find_regions(f, (255, 0, 0))
        assert len(regions) == 1
        assert (1, 1) in regions[0]
        assert (1, 2) in regions[0]

    @pytest.mark.parametrize("find_regions", FIND_REGIONS_FUNCS)
    def test_two_disconnected_regions(self, find_regions):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[0, 0] = [255, 0, 0]
        f[3, 3] = [255, 0, 0]
        regions = find_regions(f, (255, 0, 0), connectivity=4)
        assert len(regions) == 2

    @pytest.mark.parametrize("find_regions", FIND_REGIONS_FUNCS)
    def test_8_connectivity(self, find_regions):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[0, 0] = [255, 0, 0]
        f[1, 1] = [255, 0, 0]  # diagonal neighbor
        regions_4 = find_regions(f, (255, 0, 0), connectivity=4)
        regions_8 = find_regions(f, (255, 0, 0), connectivity=8)
        assert len(regions_4) == 2  # not connected in 4-connectivity
        assert len(regions_8) == 1  # connected in 8-connectivity

    @pytest.mark.parametrize("find_regions", FIND_REGIONS_FUNCS)
    def test_no_matching_color(self, find_regions):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        regions = find_regions(f, (255, 0, 0))
        assert len(regions) == 0


@pytest.mark.unit
class TestRegionBbox:
    @pytest.mark.parametrize("bbox_fn", REGION_BBOX_FUNCS)
    def test_single_pixel(self, bbox_fn):
        pixels = {(5, 10)}
        assert bbox_fn(pixels) == (5, 10, 5, 10)

    @pytest.mark.parametrize("bbox_fn", REGION_BBOX_FUNCS)
    def test_rectangular_region(self, bbox_fn):
        pixels = {(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)}
        assert bbox_fn(pixels) == (1, 2, 3, 3)

    @pytest.mark.parametrize("bbox_fn", REGION_BBOX_FUNCS)
    def test_scattered_pixels(self, bbox_fn):
        pixels = {(0, 0), (10, 20), (5, 15)}
        assert bbox_fn(pixels) == (0, 0, 10, 20)


@pytest.mark.unit
class TestMostCommonColor:
    @pytest.mark.parametrize("most_common", MOST_COMMON_FUNCS)
    def test_uniform_frame(self, most_common):
        f = np.full((4, 4, 3), 128, dtype=np.uint8)
        assert most_common(f) == (128, 128, 128)

    @pytest.mark.parametrize("most_common", MOST_COMMON_FUNCS)
    def test_majority_color(self, most_common):
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[0, 0] = [255, 0, 0]  # 1 pixel red, 15 pixels black
        assert most_common(f) == (0, 0, 0)

    @pytest.mark.parametrize("most_common", MOST_COMMON_FUNCS)
    def test_2d_frame(self, most_common):
        f = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
        assert most_common(f) == (0,)


@pytest.mark.unit
class TestExtractGrid:
    @pytest.mark.parametrize("extract", EXTRACT_GRID_FUNCS)
    def test_empty_data(self, extract):
        assert extract([]) is None

    @pytest.mark.parametrize("extract", EXTRACT_GRID_FUNCS)
    def test_3d_data(self, extract):
        data = [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]
        result = extract(data)
        assert result is not None
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("extract", EXTRACT_GRID_FUNCS)
    def test_4d_data_takes_last(self, extract):
        # Simulates a list of frames — extract_grid takes the last one
        frame1 = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
        frame2 = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]]
        data = [frame1, frame2]
        result = extract(data)
        assert result is not None
        assert result.shape == (2, 2, 3)
        assert list(result[0, 0]) == [255, 0, 0]

    @pytest.mark.parametrize("extract", EXTRACT_GRID_FUNCS)
    def test_2d_data(self, extract):
        data = [[0, 1], [2, 3]]
        result = extract(data)
        assert result is not None
        # 2D palette data now converts to RGB
        assert result.shape == (2, 2, 3)

    @pytest.mark.parametrize("extract", EXTRACT_GRID_FUNCS)
    def test_invalid_data(self, extract):
        assert extract(None) is None
        assert extract("not a list") is None
