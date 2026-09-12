from pathlib import Path
from unittest.mock import patch

from app.models.schema import VideoParams
from app.services import task


def test_generate_final_videos_bypasses_stock_clip_splitting_for_cartoon(tmp_path):
    task_id = "cartoon-final"
    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True)
    cartoon = task_dir / "cartoon-material.mp4"
    cartoon.write_bytes(b"complete-timeline")
    audio = task_dir / "audio.mp3"
    audio.write_bytes(b"audio")
    subtitle = task_dir / "subtitle.srt"
    subtitle.write_text("", encoding="utf-8")
    params = VideoParams(
        video_subject="cartoon",
        video_script="A complete timeline.",
        video_source="ai_cartoon",
        video_count=1,
        video_clip_duration=3,
        cartoon_ai_director=False,
    )

    def fake_task_dir(value=""):
        assert not value or value == task_id
        return str(task_dir if value else tmp_path)

    def fake_generate_video(**kwargs):
        # This assertion is the regression guard: the full rendered material must be
        # copied through byte-for-byte rather than reduced to the first 3-second clip.
        assert Path(kwargs["video_path"]).read_bytes() == b"complete-timeline"
        Path(kwargs["output_file"]).write_bytes(b"final")
        return True

    with (
        patch("app.services.task.utils.task_dir", side_effect=fake_task_dir),
        patch("app.services.task.video.combine_videos") as combine,
        patch("app.services.task.video.generate_video", side_effect=fake_generate_video),
        patch("app.services.task.sm.state.update_task"),
    ):
        final_paths, combined_paths, warnings = task.generate_final_videos(
            task_id=task_id,
            params=params,
            downloaded_videos=[str(cartoon)],
            audio_file=str(audio),
            subtitle_path=str(subtitle),
            audio_duration=9.0,
        )

    combine.assert_not_called()
    assert warnings == []
    assert Path(combined_paths[0]).read_bytes() == b"complete-timeline"
    assert Path(final_paths[0]).read_bytes() == b"final"
