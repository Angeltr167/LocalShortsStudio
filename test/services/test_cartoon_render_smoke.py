from pathlib import Path

import pytest

from app.services import cartoon_engine


@pytest.mark.parametrize(
    "overlay",
    sorted(cartoon_engine._OVERLAYS - {"none"}),
)
def test_every_graphic_overlay_renders_without_error(overlay):
    scene = cartoon_engine.CartoonScene(
        scene=1,
        start=0.0,
        end=1.0,
        narration="Explain the idea clearly.",
        layout="graphic",
        speaker="host",
        emotion="neutral",
        action="explain",
        overlay=overlay,
        overlay_text="$10,800" if overlay in {"stat", "money"} else "CLEAR IDEA",
        accent_text="CLEAR IDEA",
    )
    renderer = cartoon_engine.DoodleRenderer(
        width=180,
        height=320,
        scenes=[scene],
        mouth_cues=[],
    )
    frame = renderer.render(0.5)
    assert frame.mode == "RGB"
    assert frame.size == (180, 320)


def test_ffmpeg_raw_frame_stream_produces_playable_material(tmp_path):
    scene = cartoon_engine.CartoonScene(
        scene=1,
        start=0.0,
        end=0.35,
        narration="A small render smoke test.",
        layout="two_shot",
        speaker="host",
        emotion="happy",
        action="explain",
        overlay="none",
    )
    renderer = cartoon_engine.DoodleRenderer(
        width=180,
        height=320,
        scenes=[scene],
        mouth_cues=[],
    )
    output = tmp_path / "cartoon-smoke.mp4"
    cartoon_engine._render_raw_frames(
        renderer=renderer,
        duration=0.35,
        fps=12,
        output_path=output,
    )
    assert output.is_file()
    assert output.stat().st_size > 1000
    assert Path(output).suffix == ".mp4"
