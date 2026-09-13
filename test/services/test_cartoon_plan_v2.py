import json
from unittest.mock import patch

import pytest

from app.services import cartoon_engine as engine


SCRIPT = "Stop working. The task stays on your mind. Write the next step. Your attention can move on."


def plan(response=None):
    with patch.object(engine.llm, "generate_cartoon_direction", return_value=response):
        return engine.build_cartoon_plan(subject="pending work", script=SCRIPT,
                                         audio_duration=10, ai_director=response is not None)


def test_fallback_contains_bounded_events_not_only_labels():
    scenes, _ = plan()
    assert engine.CARTOON_PLAN_VERSION == 2
    assert [s.scene_template for s in scenes] == [
        "stop_work", "mental_persistence", "write_next_step", "mental_release"
    ]
    assert all(s.visual_action and s.state_before != s.state_after for s in scenes)
    assert all(s.continuity_object == "pending_item_1" for s in scenes)


@pytest.mark.parametrize("raw", [
    {"scene": 1, "scene_template": "execute_python", "state_after": "completed"},
    {"scene": 1, "scene_template": "task_parked", "state_before": "neutral", "state_after": "completed"},
    {"scene": 1, "start": -100, "end": 1000, "narration": "overwritten", "draw": "import os"},
])
def test_invalid_or_freeform_choices_cannot_change_timeline_or_create_states(raw):
    baseline, _ = plan()
    scenes, _ = plan(json.dumps({"scenes": [raw]}))
    assert len(scenes) == len(baseline)
    assert [(s.start, s.end, s.narration) for s in scenes] == [
        (s.start, s.end, s.narration) for s in baseline
    ]
    assert scenes[0].state_after != "completed"
    assert "draw" not in scenes[0].to_dict()


def test_duplicate_scene_numbers_fall_back_instead_of_overwriting():
    scenes, director = plan('{"scenes":[{"scene":1},{"scene":1}]}')
    assert director == "deterministic_fallback"
    assert [s.to_dict() for s in scenes] == [s.to_dict() for s in plan()[0]]


def test_missing_scenes_are_locally_filled():
    scenes, _ = plan('{"scenes":[{"scene":1,"scene_template":"stop_work"}]}')
    assert len(scenes) == 4
    assert scenes[-1].scene_template == "mental_release"


def test_unrelated_topic_and_negation_do_not_invent_causal_story():
    for script in ("A browser supports many features.", "Do not write a next step."):
        scenes, _ = engine.build_cartoon_plan(subject="other", script=script,
                                              audio_duration=4, ai_director=False)
        assert scenes[0].scene_template == "explain_generic"
        assert scenes[0].continuity_object == ""


def test_second_scenario_can_record_an_idea_without_fixture_phrasing():
    scenes, _ = engine.build_cartoon_plan(
        subject="ideas", script="Record your idea on paper. Save it for later.",
        audio_duration=5, ai_director=False,
    )
    assert [s.scene_template for s in scenes] == ["write_next_step", "task_parked"]


def test_prompt_requests_neighbors_events_and_bounded_states():
    prompt = engine._director_prompt("work", plan()[0])
    assert "previous" in prompt and "next" in prompt
    assert "state_before" in prompt and "state_after" in prompt
    assert "coordinates" in prompt
