import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.models.schema import MaterialInfo, VideoParams
from app.services import gpt_visual_review, semantic_ranker, video


def _task_dir_patch(tmp_path):
    def task_dir(task_id=None):
        root = tmp_path / "tasks"
        root.mkdir(parents=True, exist_ok=True)
        if task_id is None:
            return str(root)
        target = root / str(task_id)
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    return task_dir


def _item(asset_id: str, score: float, url_suffix: str = "") -> MaterialInfo:
    return MaterialInfo(
        provider="pexels",
        url=f"https://videos.pexels.com/video-files/{asset_id}{url_suffix}.mp4",
        duration=8,
        source_info={
            "provider": "pexels",
            "asset_id": asset_id,
            "preview_images": [],
            "semantic_score": score,
            "semantic_positive_score": score + 0.1,
            "semantic_rank": 1,
        },
    )


def _write_script(task_dir: Path):
    params = VideoParams(
        video_subject="unfinished work",
        video_source="pexels",
        strict_scene_matching=True,
        semantic_scene_ranking=True,
        video_clip_duration=3,
    )
    (task_dir / "script.json").write_text(
        json.dumps(
            {
                "script": "unfinished task narration",
                "search_terms": ["unfinished computer work"],
                "params": params.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )


def test_record_scene_candidates_creates_private_registry(tmp_path):
    task_id = "task-review"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        task_dir = Path(gpt_visual_review._safe_task_directory(task_id))
        _write_script(task_dir)
        selected_file = task_dir / "selected.mp4"
        selected_file.write_bytes(b"video")
        first = _item("first", 0.61)
        second = _item("second", 0.59)

        assert gpt_visual_review.record_scene_candidates(
            task_id=task_id,
            scene={
                "scene": 1,
                "start": 0,
                "end": 3,
                "duration": 3,
                "narration_text": "unfinished task stays in your head",
                "query": "unfinished computer work",
                "semantic_query": "unfinished computer work at a laptop",
                "reused": False,
                "bridge_selected": False,
            },
            ranked_items=[first, second],
            selected_item=first,
            selected_path=str(selected_file),
            selected_query="unfinished computer work",
        ) is True

        registry = gpt_visual_review.load_registry(task_id)
        scene = registry["scenes"][0]
        assert scene["current_candidate_id"] == "S01-C1"
        assert scene["candidates"][0]["download_url"].startswith("https://videos.pexels.com/")
        assert scene["review_required"] is True
        assert "narrow_top_score_margin" in scene["review_reasons"]


def test_export_package_strips_private_urls_and_builds_contact_sheet(tmp_path):
    task_id = "task-export"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        task_dir = gpt_visual_review._safe_task_directory(task_id)
        _write_script(task_dir)
        (task_dir / "final-1.mp4").write_bytes(b"not-a-real-video-but-export-only")
        selected_file = task_dir / "selected.mp4"
        selected_file.write_bytes(b"video")
        first = _item("first", 0.40)
        second = _item("second", 0.39)
        gpt_visual_review.record_scene_candidates(
            task_id=task_id,
            scene={
                "scene": 1,
                "start": 0,
                "end": 3,
                "duration": 3,
                "narration_text": "unfinished browser work",
                "query": "multiple browser tabs unfinished work",
            },
            ranked_items=[first, second],
            selected_item=first,
            selected_path=str(selected_file),
            selected_query="multiple browser tabs unfinished work",
        )

        package = gpt_visual_review.export_review_package(task_id, scope="all")
        zip_path = Path(package["zip_path"])
        assert zip_path.is_file()
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            assert "review_manifest.json" in names
            assert "scene_01.jpg" in names
            assert "current_video.mp4" in names
            manifest_text = archive.read("review_manifest.json").decode("utf-8")
            manifest = json.loads(manifest_text)
        assert "download_url" not in manifest_text
        assert "preview_urls" not in manifest_text
        assert manifest["task_id"] == task_id
        assert manifest["scenes"][0]["candidates"][0]["candidate_id"] == "S01-C1"


def test_validate_decisions_rejects_invented_candidate_id(tmp_path):
    task_id = "task-validate"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        task_dir = gpt_visual_review._safe_task_directory(task_id)
        _write_script(task_dir)
        selected_file = task_dir / "selected.mp4"
        selected_file.write_bytes(b"video")
        first = _item("first", 0.4)
        second = _item("second", 0.3)
        gpt_visual_review.record_scene_candidates(
            task_id=task_id,
            scene={"scene": 1, "start": 0, "end": 3, "duration": 3, "query": "work"},
            ranked_items=[first, second],
            selected_item=first,
            selected_path=str(selected_file),
            selected_query="work",
        )
        gpt_visual_review.export_review_package(task_id, scope="all")

        with pytest.raises(gpt_visual_review.VisualReviewError, match="unknown candidate_id"):
            gpt_visual_review.validate_decisions(
                task_id,
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "decisions": [
                        {
                            "scene": 1,
                            "action": "select",
                            "candidate_id": "S01-C99",
                            "confidence": 0.9,
                            "reason": "invented",
                        }
                    ],
                },
            )


def test_validate_retry_search_rejects_urls(tmp_path):
    task_id = "task-retry"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        task_dir = gpt_visual_review._safe_task_directory(task_id)
        _write_script(task_dir)
        selected_file = task_dir / "selected.mp4"
        selected_file.write_bytes(b"video")
        first = _item("first", 0.3)
        second = _item("second", 0.29)
        gpt_visual_review.record_scene_candidates(
            task_id=task_id,
            scene={"scene": 1, "start": 0, "end": 3, "duration": 3, "query": "work"},
            ranked_items=[first, second],
            selected_item=first,
            selected_path=str(selected_file),
            selected_query="work",
        )
        gpt_visual_review.export_review_package(task_id, scope="all")
        with pytest.raises(gpt_visual_review.VisualReviewError, match="short visible-footage query"):
            gpt_visual_review.validate_decisions(
                task_id,
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "decisions": [
                        {
                            "scene": 1,
                            "action": "retry_search",
                            "confidence": 0.9,
                            "search_query": "https://example.com/bad",
                            "avoid": [],
                            "reason": "bad",
                        }
                    ],
                },
            )


def test_semantic_ranker_prioritizes_manual_negative_queries():
    first = _item("first", 0.4)
    second = _item("second", 0.3)
    first.source_info["preview_images"] = ["https://images.pexels.com/first.jpg"]
    second.source_info["preview_images"] = ["https://images.pexels.com/second.jpg"]
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model": "test",
        "ranked": [
            {"id": "pexels:first", "score": 0.4},
            {"id": "pexels:second", "score": 0.3},
        ],
    }
    session = Mock()
    session.post.return_value = response
    with patch("app.services.semantic_ranker._local_session", return_value=session):
        semantic_ranker.rank_materials(
            "multiple browser tabs unfinished work",
            [first, second],
            enabled=True,
            extra_negative_queries=["startup screen", "empty office"],
        )
    payload = session.post.call_args.kwargs["json"]
    assert payload["negative_queries"][:2] == ["startup screen", "empty office"]


def test_rebuild_reuses_master_audio_and_existing_subtitles(tmp_path):
    task_id = "task-rebuild"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        task_dir = gpt_visual_review._safe_task_directory(task_id)
        _write_script(task_dir)
        original = task_dir / "final-1.mp4"
        original.write_bytes(b"original")
        subtitle = task_dir / "subtitle.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        scene_file = task_dir / "scene.mp4"
        scene_file.write_bytes(b"scene")
        registry = {
            "schema": gpt_visual_review.REGISTRY_SCHEMA,
            "schema_version": 1,
            "task_id": task_id,
            "package_revision": 1,
            "review_revision": 0,
            "scenes": [
                {
                    "scene": 1,
                    "current_path": str(scene_file),
                    "current_candidate_id": "S01-C1",
                    "candidates": [],
                }
            ],
        }
        gpt_visual_review._write_json_atomic(gpt_visual_review._registry_path(task_id), registry)

        def fake_extract(source, destination):
            assert source == original
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"mixed-master-audio")

        def fake_combine(**kwargs):
            Path(kwargs["combined_video_path"]).write_bytes(b"combined")
            assert kwargs["video_paths"] == [str(scene_file)]

        def fake_generate(**kwargs):
            assert kwargs["subtitle_path"] == str(subtitle)
            assert kwargs["params"].voice_volume == 1.0
            assert kwargs["params"].bgm_type == ""
            Path(kwargs["output_file"]).write_bytes(b"reviewed")
            return True

        with (
            patch("app.services.gpt_visual_review._extract_master_audio", side_effect=fake_extract),
            patch.object(video, "combine_videos", side_effect=fake_combine),
            patch.object(video, "generate_video", side_effect=fake_generate),
        ):
            rebuilt = gpt_visual_review._rebuild_reviewed_video(task_id, registry)

        assert Path(rebuilt).name == "final-reviewed-1.mp4"
        assert Path(rebuilt).read_bytes() == b"reviewed"
