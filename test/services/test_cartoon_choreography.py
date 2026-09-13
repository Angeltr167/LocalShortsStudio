import pytest

from app.services import cartoon_choreography as choreography
from app.services.cartoon_engine import CartoonScene
from app.services import cartoon_engine as engine


def make_scene(duration):
    return CartoonScene(1, 0, duration, "Action", "host", "host", "neutral", "explain", "none")


def test_long_scene_visits_all_ordered_phases():
    names = [choreography.phase_at(i / 100, 5).name for i in range(100)]
    assert names[0] == "enter"
    assert names[-1] == "resolve"
    assert [name for name in choreography.PHASE_NAMES if name in names] == list(choreography.PHASE_NAMES)


def test_short_scene_compresses_setup_but_keeps_action_and_resolution():
    assert choreography.phase_at(0.01, 0.4).name == "enter"
    assert choreography.phase_at(0.5, 0.4).name == "action"
    assert choreography.phase_at(0.99, 0.4).name == "resolve"


@pytest.mark.parametrize("duration", [0.01, 0.4, 1.6, 6.5])
def test_phase_is_bounded_and_never_divides_by_zero(duration):
    for progress in (-1, 0, 0.2, 0.8, 1, 2):
        phase = choreography.phase_at(progress, duration)
        assert phase.name in choreography.PHASE_NAMES
        assert 0 <= phase.progress <= 1


def test_stage_at_uses_absolute_time_and_is_repeatable():
    scene = make_scene(4)
    assert choreography.stage_at(scene, 0).name == "enter"
    assert choreography.stage_at(scene, 3.99).name == "resolve"
    assert choreography.stage_at(scene, 2) == choreography.stage_at(scene, 2)


def test_supported_story_scene_changes_between_action_and_resolution():
    cue = engine.NarrationCue(0, 4, "Write the next step and save it for later.")
    scene = engine._fallback_scene(cue, 0)
    renderer = engine.DoodleRenderer(180, 320, [scene], [])
    action = renderer.render(1.8)
    resolved = renderer.render(3.8)
    assert action.tobytes() != resolved.tobytes()
