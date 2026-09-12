import json
from pathlib import Path
from unittest.mock import patch

from app.models.schema import VideoParams
from app.services import cartoon_engine


def test_parse_srt_cues(tmp_path):
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nYour brain keeps a task open.\n\n"
        "2\n00:00:02,000 --> 00:00:04,600\nAlmost like a browser tab.\n",
        encoding="utf-8",
    )
    cues = cartoon_engine.parse_srt_cues(subtitle)
    assert len(cues) == 2
    assert cues[0].start == 0
    assert cues[1].end == 4.6
    assert "browser tab" in cues[1].text


def test_fallback_plan_maps_semantics_to_explanatory_overlays():
    script = (
        "Your brain remembers unfinished work. "
        "A half-written email can stay in your mind. "
        "Write down the next step on a checklist."
    )
    scenes, director = cartoon_engine.build_cartoon_plan(
        subject="unfinished tasks",
        script=script,
        audio_duration=9.0,
        ai_director=False,
    )
    assert director == "deterministic"
    overlays = {scene.overlay for scene in scenes}
    assert overlays & {"brain", "email", "checklist"}
    assert scenes[0].start == 0
    assert scenes[-1].end == 9.0


def test_ai_director_is_bounded_to_allowed_scene_vocabulary():
    response = json.dumps(
        {
            "scenes": [
                {
                    "scene": 1,
                    "layout": "impossible_camera",
                    "speaker": "alien",
                    "emotion": "copyrighted_character",
                    "action": "explode",
                    "overlay": "browser_tabs",
                    "overlay_text": "VISIBLE TABS",
                    "accent_text": "unfinished work",
                }
            ]
        }
    )
    with patch("app.services.cartoon_engine.llm.generate_cartoon_direction", return_value=response):
        scenes, director = cartoon_engine.build_cartoon_plan(
            subject="unfinished work",
            script="A browser tab stays mentally open.",
            audio_duration=3.0,
            ai_director=True,
        )
    assert director == "ai"
    scene = scenes[0]
    assert scene.layout in cartoon_engine._LAYOUTS
    assert scene.speaker in cartoon_engine._SPEAKERS
    assert scene.emotion in cartoon_engine._EMOTIONS
    assert scene.action in cartoon_engine._ACTIONS
    assert scene.overlay == "browser_tabs"
    assert scene.overlay_text == "VISIBLE TABS"


def test_invalid_ai_director_response_fails_open_to_deterministic_plan():
    with patch(
        "app.services.cartoon_engine.llm.generate_cartoon_direction",
        return_value="not-json",
    ):
        scenes, director = cartoon_engine.build_cartoon_plan(
            subject="focus",
            script="Notifications pull your attention away.",
            audio_duration=3.0,
            ai_director=True,
        )
    assert director == "deterministic_fallback"
    assert scenes
    assert scenes[0].overlay == "phone"


def test_doodle_renderer_produces_full_rgb_frame():
    cue = cartoon_engine.NarrationCue(0.0, 3.0, "Write the next step on a checklist.")
    scene = cartoon_engine._fallback_scene(cue, 0)
    renderer = cartoon_engine.DoodleRenderer(
        width=360,
        height=640,
        scenes=[scene],
        mouth_cues=[],
    )
    frame = renderer.render(1.0)
    assert frame.mode == "RGB"
    assert frame.size == (360, 640)
    # The renderer should not output a flat frame; sample distant pixels.
    assert frame.getpixel((10, 10)) != frame.getpixel((180, 500))


def test_rhubarb_payload_parser_rejects_unknown_shapes():
    cues = cartoon_engine._parse_rhubarb_payload(
        {
            "mouthCues": [
                {"start": 0.0, "end": 0.2, "value": "A"},
                {"start": 0.2, "end": 0.4, "value": "unknown"},
            ]
        }
    )
    assert [cue.value for cue in cues] == ["A", "X"]


def test_render_cartoon_material_writes_plan_storyboard_and_material(tmp_path):
    task_id = "cartoon-test"
    task_dir = tmp_path / "tasks" / task_id
    task_dir.mkdir(parents=True)
    audio = task_dir / "audio.mp3"
    audio.write_bytes(b"fake-audio")
    subtitle = task_dir / "subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nA browser tab stays open.\n",
        encoding="utf-8",
    )
    params = VideoParams(
        video_subject="unfinished work",
        video_script="A browser tab stays open.",
        video_source="ai_cartoon",
        cartoon_ai_director=False,
        cartoon_lip_sync="heuristic",
        cartoon_fps=18,
    )

    def fake_task_dir(sub_dir=""):
        root = tmp_path / "tasks"
        root.mkdir(parents=True, exist_ok=True)
        if sub_dir:
            target = root / str(sub_dir)
            target.mkdir(parents=True, exist_ok=True)
            return str(target)
        return str(root)

    def fake_render(**kwargs):
        output = kwargs["output_path"]
        output.write_bytes(b"fake-cartoon-video")

    with (
        patch("app.services.cartoon_engine.utils.task_dir", side_effect=fake_task_dir),
        patch("app.services.cartoon_engine._render_raw_frames", side_effect=fake_render),
        patch("app.services.cartoon_engine.task_artifacts.patch_script_data"),
    ):
        result = cartoon_engine.render_cartoon_material(
            task_id=task_id,
            params=params,
            video_script=params.video_script,
            audio_file=str(audio),
            audio_duration=2.0,
            subtitle_path=str(subtitle),
        )

    assert Path(result).read_bytes() == b"fake-cartoon-video"
    plan = json.loads((task_dir / "cartoon_plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == cartoon_engine.CARTOON_PLAN_SCHEMA
    assert plan["director"] == "deterministic"
    assert plan["lip_sync"] == "heuristic"
    assert plan["scenes"]
    assert (task_dir / "cartoon_storyboard.jpg").is_file()
