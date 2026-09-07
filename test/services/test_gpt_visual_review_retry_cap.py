import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.models.schema import MaterialInfo, VideoParams
from app.services import gpt_visual_review


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


def _write_script(task_dir: Path):
    task_dir.mkdir(parents=True, exist_ok=True)
    params = VideoParams(
        video_subject="unfinished work",
        video_source="pexels",
        strict_scene_matching=True,
        semantic_scene_ranking=True,
        video_clip_duration=3,
    )
    (task_dir / "script.json").write_text(
        json.dumps({"params": params.model_dump(mode="json")}),
        encoding="utf-8",
    )


def _candidate(candidate_id: str, asset_id: str):
    return {
        "candidate_id": candidate_id,
        "provider": "pexels",
        "asset_id": asset_id,
        "duration": 8.0,
        "download_url": f"https://videos.pexels.com/{asset_id}.mp4",
        "preview_urls": [],
        "local_path": "",
    }


def _write_registry(task_id: str, *, retry_round: int, tmp_path: Path):
    task_dir = gpt_visual_review._safe_task_directory(task_id)
    _write_script(task_dir)
    registry = {
        "schema": gpt_visual_review.REGISTRY_SCHEMA,
        "schema_version": 1,
        "task_id": task_id,
        "package_revision": 1,
        "review_revision": 0,
        "last_package": {
            "revision": 1,
            "scope": "retry",
            "zip_path": "",
            "included_scenes": [7],
        },
        "scenes": [
            {
                "scene": 7,
                "start": 18.0,
                "end": 21.0,
                "duration": 3.0,
                "narration": "almost like a tab waiting to be closed",
                "visual_intent": "multiple open browser tabs unfinished work computer",
                "current_candidate_id": "S07-C1",
                "current_path": str(task_dir / "current.mp4"),
                "candidates": [
                    _candidate("S07-C1", "current"),
                    _candidate("S07-R1-C1", "retry1"),
                ],
                "retry_round": retry_round,
                "retry_pending": True,
                "review_required": True,
            }
        ],
    }
    (task_dir / "current.mp4").write_bytes(b"video")
    gpt_visual_review._write_json_atomic(gpt_visual_review._registry_path(task_id), registry)
    return registry


def test_retry_search_is_rejected_after_hard_limit(tmp_path):
    task_id = "retry-limit"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        _write_registry(
            task_id,
            retry_round=gpt_visual_review.MAX_RETRY_ROUNDS,
            tmp_path=tmp_path,
        )
        with pytest.raises(gpt_visual_review.VisualReviewError, match="exhausted"):
            gpt_visual_review.validate_decisions(
                task_id,
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "decisions": [
                        {
                            "scene": 7,
                            "action": "retry_search",
                            "confidence": 0.99,
                            "search_query": "close up browser with multiple tabs",
                            "avoid": ["startup screen"],
                            "reason": "literal match still missing",
                        }
                    ],
                },
            )


def test_select_remains_valid_after_retry_limit(tmp_path):
    task_id = "retry-select"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        _write_registry(
            task_id,
            retry_round=gpt_visual_review.MAX_RETRY_ROUNDS,
            tmp_path=tmp_path,
        )
        decisions = gpt_visual_review.validate_decisions(
            task_id,
            {
                "schema_version": 1,
                "task_id": task_id,
                "decisions": [
                    {
                        "scene": 7,
                        "action": "select",
                        "candidate_id": "S07-R1-C1",
                        "confidence": 0.8,
                        "reason": "best understandable fallback",
                    }
                ],
            },
        )
        assert decisions[0]["action"] == "select"


def test_manifest_exposes_retry_budget(tmp_path):
    task_id = "retry-manifest"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        registry = _write_registry(
            task_id,
            retry_round=gpt_visual_review.MAX_RETRY_ROUNDS,
            tmp_path=tmp_path,
        )
        scene = gpt_visual_review._manifest_scene(registry["scenes"][0], included=True)
        assert scene["retry_round"] == gpt_visual_review.MAX_RETRY_ROUNDS
        assert scene["max_retry_rounds"] == gpt_visual_review.MAX_RETRY_ROUNDS
        assert scene["retry_exhausted"] is True


def test_retry_refresh_preserves_current_candidate(tmp_path):
    task_id = "retry-preserve"
    with patch("app.services.gpt_visual_review.utils.task_dir", side_effect=_task_dir_patch(tmp_path)):
        registry = _write_registry(task_id, retry_round=1, tmp_path=tmp_path)
        scene = registry["scenes"][0]
        params = VideoParams(
            video_subject="unfinished work",
            video_source="pexels",
            strict_scene_matching=True,
            semantic_scene_ranking=True,
            video_clip_duration=3,
        )
        ranked = [
            MaterialInfo(
                provider="pexels",
                url=f"https://videos.pexels.com/new-{index}.mp4",
                duration=8,
                source_info={"provider": "pexels", "asset_id": f"new-{index}"},
            )
            for index in range(1, 5)
        ]
        with (
            patch("app.services.gpt_visual_review._provider_search", return_value=ranked),
            patch("app.services.semantic_ranker.rank_materials", return_value=ranked),
        ):
            gpt_visual_review._refresh_retry_scene(
                task_id,
                scene,
                query="close up browser tabs",
                avoid=["startup screen"],
                params=params,
            )
        assert scene["current_candidate_id"] == "S07-C1"
        assert scene["candidates"][0]["candidate_id"] == "S07-C1"
        assert len(scene["candidates"]) == gpt_visual_review.MAX_REVIEW_CANDIDATES
        assert scene["retry_round"] == 2
