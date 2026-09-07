from pathlib import Path
from unittest.mock import patch

from app.models.schema import MaterialInfo, VideoParams
from app.services import material, strict_scene, task


def _item(url: str, asset_id: str, search_term: str) -> MaterialInfo:
    return MaterialInfo(
        provider="pexels",
        url=url,
        duration=9,
        source_info={
            "provider": "pexels",
            "asset_id": asset_id,
            "search_term": search_term,
        },
    )


def test_strict_scene_matching_defaults_off():
    params = VideoParams(video_subject="test")
    assert params.strict_scene_matching is False


def test_scene_plan_repeats_queries_only_in_contiguous_slots():
    plan = strict_scene.build_scene_plan(
        ["first action", "second action"],
        audio_duration=7,
        max_clip_duration=3,
    )

    assert [scene["query"] for scene in plan] == [
        "first action",
        "first action",
        "second action",
    ]
    assert [(scene["start"], scene["end"]) for scene in plan] == [
        (0, 3),
        (3, 6),
        (6, 7.0),
    ]


def test_strict_downloader_avoids_duplicate_asset_when_alternative_exists():
    shared = _item("https://example.test/shared.mp4", "shared", "one")
    unique = _item("https://example.test/unique.mp4", "unique", "two")

    def search_videos(search_term, minimum_duration, video_aspect):
        del minimum_duration, video_aspect
        if search_term == "one":
            return [shared]
        return [shared, unique]

    saved_plans = []

    def capture_plan(task_id, **updates):
        del task_id
        if "scene_plan" in updates:
            saved_plans.append([dict(scene) for scene in updates["scene_plan"]])
        return True

    with patch(
        "app.services.strict_scene.task_artifacts.patch_script_data",
        side_effect=capture_plan,
    ):
        paths = strict_scene.download_videos_by_scene_queries(
            task_id="strict-test",
            search_terms=["one", "two"],
            search_videos=search_videos,
            save_video=lambda video_url, save_dir: str(
                Path(save_dir or "/tmp") / Path(video_url).name
            ),
            source_record=lambda item, path: {
                "asset_id": item.source_info["asset_id"],
                "local_file": Path(path).name,
            },
            persist_sources=lambda task_id, sources: None,
            redact_error=lambda error, secret: str(error).replace(secret, "***"),
            video_aspect=material.VideoAspect.portrait,
            audio_duration=6,
            max_clip_duration=3,
            material_directory="/tmp",
        )

    assert [Path(path).name for path in paths] == [
        "shared.mp4",
        "unique.mp4",
    ]
    assert saved_plans[-1][0]["reused"] is False
    assert saved_plans[-1][1]["reused"] is False


def test_strict_downloader_reuses_only_as_last_resort():
    shared = _item("https://example.test/shared.mp4", "shared", "one")
    saved_plans = []

    def search_videos(search_term, minimum_duration, video_aspect):
        del search_term, minimum_duration, video_aspect
        return [shared]

    def capture_plan(task_id, **updates):
        del task_id
        if "scene_plan" in updates:
            saved_plans.append([dict(scene) for scene in updates["scene_plan"]])
        return True

    with patch(
        "app.services.strict_scene.task_artifacts.patch_script_data",
        side_effect=capture_plan,
    ):
        paths = strict_scene.download_videos_by_scene_queries(
            task_id="strict-test",
            search_terms=["one", "two"],
            search_videos=search_videos,
            save_video=lambda video_url, save_dir: "/tmp/shared.mp4",
            source_record=lambda item, path: {},
            persist_sources=lambda task_id, sources: None,
            redact_error=lambda error, secret: str(error).replace(secret, "***"),
            video_aspect=material.VideoAspect.portrait,
            audio_duration=6,
            max_clip_duration=3,
            material_directory="/tmp",
        )

    assert paths == ["/tmp/shared.mp4", "/tmp/shared.mp4"]
    assert saved_plans[-1][0]["reused"] is False
    assert saved_plans[-1][1]["reused"] is True


def test_existing_ordered_path_is_unchanged_when_strict_is_off():
    with patch(
        "app.services.material._download_videos_by_script_order",
        return_value=["ordered.mp4"],
    ) as ordered:
        result = material.download_videos(
            task_id="ordered-test",
            search_terms=["one"],
            source="pexels",
            audio_duration=3,
            max_clip_duration=3,
            match_script_order=True,
            strict_scene_matching=False,
        )

    assert result == ["ordered.mp4"]
    ordered.assert_called_once()


def test_strict_stock_path_is_opt_in():
    with patch(
        "app.services.strict_scene.download_videos_by_scene_queries",
        return_value=["strict.mp4"],
    ) as strict:
        result = material.download_videos(
            task_id="strict-test",
            search_terms=["one"],
            source="pexels",
            audio_duration=3,
            max_clip_duration=3,
            match_script_order=False,
            strict_scene_matching=True,
        )

    assert result == ["strict.mp4"]
    strict.assert_called_once()


def test_task_strict_scene_matching_is_stock_only():
    stock_params = VideoParams(
        video_subject="test",
        video_source="pexels",
        strict_scene_matching=True,
    )
    ai_params = VideoParams(
        video_subject="test",
        video_source="wavespeed",
        strict_scene_matching=True,
    )
    local_params = VideoParams(
        video_subject="test",
        video_source="local",
        strict_scene_matching=True,
    )

    assert task._strict_scene_matching_enabled(stock_params) is True
    assert task._strict_scene_matching_enabled(ai_params) is False
    assert task._strict_scene_matching_enabled(local_params) is False
