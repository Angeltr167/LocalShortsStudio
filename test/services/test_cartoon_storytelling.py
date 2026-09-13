from pathlib import Path

from app.services import cartoon_engine as engine


def renderer(script=None):
    if script is None:
        script = (Path(__file__).parents[1] / "fixtures/cartoon_storytelling/unfinished_task.txt").read_text(encoding="utf-8")
    scenes, _ = engine.build_cartoon_plan(subject="story", script=script,
                                         audio_duration=22, ai_director=False)
    return engine.DoodleRenderer(360, 640, scenes, []), scenes


def test_persistent_object_survives_every_representation():
    render, scenes = renderer()
    states = [render.story_state(scene.end - 0.000001) for scene in scenes]
    assert {s.object_id for s in states} == {"pending_item_1"}
    assert states[-2].semantic_state == "parked"
    assert states[-1].semantic_state == "released"
    assert states[-1].parked and not states[-1].loop_active
    assert all(s.semantic_state != "completed" for s in states)


def test_token_position_is_continuous_at_scene_boundaries():
    render, scenes = renderer()
    for scene in scenes[1:]:
        before = render.story_state(scene.start - 1e-7)
        after = render.story_state(scene.start)
        assert abs(before.x - after.x) < 0.01
        assert abs(before.y - after.y) < 0.01


def test_render_is_independent_of_frame_request_order():
    render, _ = renderer()
    expected = render.render(19).tobytes()
    render.render(1)
    render.render(21)
    assert render.render(19).tobytes() == expected


def test_recording_and_parking_idea_changes_visible_object_region():
    render, scenes = renderer("Record your idea on paper. Save it for later.")
    assert scenes[-1].scene_template == "task_parked"
    start = render.render(scenes[-1].start + 0.05).crop((0, 0, 360, 320))
    end = render.render(scenes[-1].end - 0.05).crop((0, 0, 360, 320))
    assert start.tobytes() != end.tobytes()
    assert render.story_state(scenes[-1].end - 0.05).parked


def test_unsupported_concept_does_not_create_pending_object():
    render, _ = renderer("The ocean is blue.")
    assert render.story_state(1).object_id == ""


def test_storyboard_includes_resolution_and_all_beats(tmp_path):
    render, scenes = renderer()
    path = tmp_path / "cartoon_storyboard.jpg"
    engine._write_storyboard(render, scenes, path)
    assert path.is_file()
    assert path.with_stem("cartoon_storyboard_02").is_file()
