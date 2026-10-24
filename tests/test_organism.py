import random

import pytest

from eidolon.organism import Symbiote, generate_appearance


def test_appearance_is_reproducible_and_seeds_vary_the_specimen() -> None:
    first = generate_appearance(17)
    assert first == generate_appearance(17)
    assert first != generate_appearance(18)
    assert first.body != generate_appearance(18).body
    assert first.tendrils != generate_appearance(18).tendrils
    assert first.core != generate_appearance(18).core


def test_appearance_generation_does_not_touch_global_random_state() -> None:
    state = random.getstate()
    generate_appearance(411)
    assert random.getstate() == state


@pytest.mark.parametrize("seed", [0, 1, -1, 17, 999, 2**64])
def test_seeded_parameters_are_bounded_and_tendrils_are_independent(seed) -> None:
    appearance = generate_appearance(seed)
    assert 7 <= len(appearance.tendrils) <= 11
    assert appearance.body.asymmetry > 0
    assert 0.18 <= appearance.core.radius <= 0.25
    assert len({t.phase for t in appearance.tendrils}) == len(appearance.tendrils)
    assert len({t.frequency for t in appearance.tendrils}) == len(appearance.tendrils)
    for tendril in appearance.tendrils:
        assert 1.15 <= tendril.length <= 2.05
        assert 0.65 <= tendril.frequency <= 1.30
    for color in (appearance.membrane_color, appearance.tissue_color, appearance.core_color):
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


def test_pose_interpolation_is_read_only_and_continuous() -> None:
    organism = Symbiote(17)
    initial = organism.pose()
    organism.update(1 / 60)
    assert organism.pose(0) == initial
    assert organism.pose(0.5).seconds == pytest.approx(1 / 120)
    assert organism.pose(1).seconds == pytest.approx(1 / 60)
    assert organism.age_seconds == pytest.approx(1 / 60)
    assert organism.previous_age_seconds == 0
    previous = organism.pose()
    organism.update(1 / 60)
    assert organism.pose(0) == previous


def test_same_seed_and_update_sequence_produce_same_motion() -> None:
    first, second = Symbiote(17), Symbiote(17)
    initial = first.pose()
    for _ in range(300):
        first.update(1 / 60)
        second.update(1 / 60)
    assert first.pose(0.4) == second.pose(0.4)
    pose = first.pose()
    assert pose.offset != initial.offset
    assert pose.rotation != initial.rotation
    assert pose.breath != initial.breath
    assert pose.core_pulse != initial.core_pulse
