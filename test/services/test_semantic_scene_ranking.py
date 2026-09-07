from pathlib import Path
from unittest.mock import Mock, patch

from app.models.schema import MaterialInfo, VideoParams
from app.services import semantic_ranker, strict_scene, task
from semantic_ranker import main as ranker_service


def _item(asset_id: str, preview: str) -> MaterialInfo:
    return MaterialInfo(
        provider="pexels",
        url=f"https://example.test/{asset_id}.mp4",
        duration=8,
        source_info={
            "provider": "pexels",
            "asset_id": asset_id,
            "preview_images": [preview],
        },
    )


def test_semantic_scene_ranking_defaults_off():
    params = VideoParams(video_subject="test")
    assert params.semantic_scene_ranking is False


def test_evenly_spaced_previews_preserve_edges():
    values = [f"https://images.pexels.com/{index}.jpg" for index in range(8)]
    assert semantic_ranker._evenly_spaced(values, 4) == [
        values[0],
        values[2],
        values[5],
        values[7],
    ]


def test_rank_materials_reorders_candidates_and_attaches_score():
    first = _item("first", "https://images.pexels.com/first.jpg")
    second = _item("second", "https://images.pexels.com/second.jpg")

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model": "ViT-B-32/test",
        "ranked": [
            {"id": "pexels:second", "score": 0.81},
            {"id": "pexels:first", "score": 0.42},
        ],
    }
    session = Mock()
    session.post.return_value = response

    with patch("app.services.semantic_ranker._local_session", return_value=session):
        ranked = semantic_ranker.rank_materials(
            "hand writing in notebook",
            [first, second],
            enabled=True,
        )

    assert [item.source_info["asset_id"] for item in ranked] == ["second", "first"]
    assert ranked[0].source_info["semantic_score"] == 0.81
    assert ranked[0].source_info["semantic_rank"] == 1
    assert ranked[0].source_info["semantic_ranker_model"] == "ViT-B-32/test"
    session.close.assert_called_once()


def test_rank_materials_falls_back_when_service_is_unavailable():
    first = _item("first", "https://images.pexels.com/first.jpg")
    second = _item("second", "https://images.pexels.com/second.jpg")
    session = Mock()
    session.post.side_effect = RuntimeError("offline")

    with patch("app.services.semantic_ranker._local_session", return_value=session):
        ranked = semantic_ranker.rank_materials("query", [first, second], enabled=True)

    assert ranked == [first, second]


def test_strict_scene_uses_semantic_candidate_order_when_enabled():
    first = _item("first", "https://images.pexels.com/first.jpg")
    second = _item("second", "https://images.pexels.com/second.jpg")

    def search_videos(search_term, minimum_duration, video_aspect):
        del search_term, minimum_duration, video_aspect
        return [first, second]

    def save_video(video_url, save_dir):
        del save_dir
        return str(Path("/tmp") / Path(video_url).name)

    with (
        patch("app.services.strict_scene.task_artifacts.patch_script_data", return_value=True),
        patch(
            "app.services.strict_scene.semantic_ranker.rank_materials",
            return_value=[second, first],
        ) as rerank,
    ):
        paths = strict_scene.download_videos_by_scene_queries(
            task_id="semantic-test",
            search_terms=["writing notebook"],
            search_videos=search_videos,
            save_video=save_video,
            source_record=lambda item, path: {"asset_id": item.source_info["asset_id"]},
            persist_sources=lambda task_id, sources: None,
            redact_error=lambda error, secret: str(error).replace(secret, "***"),
            video_aspect=ranker_service.VideoAspect.portrait if hasattr(ranker_service, "VideoAspect") else __import__("app.models.schema", fromlist=["VideoAspect"]).VideoAspect.portrait,
            audio_duration=3,
            max_clip_duration=3,
            material_directory="/tmp",
            semantic_scene_ranking=True,
        )

    assert Path(paths[0]).name == "second.mp4"
    rerank.assert_called_once()


def test_task_semantic_ranking_requires_strict_stock_source():
    params = VideoParams(
        video_subject="test",
        video_source="pexels",
        strict_scene_matching=True,
        semantic_scene_ranking=True,
    )
    assert task._semantic_scene_ranking_enabled(params) is True

    params.strict_scene_matching = False
    assert task._semantic_scene_ranking_enabled(params) is False

    params.strict_scene_matching = True
    params.video_source = "local"
    assert task._semantic_scene_ranking_enabled(params) is False


def test_ranker_service_accepts_only_known_https_preview_hosts():
    assert ranker_service._allowed_preview_url(
        "https://images.pexels.com/videos/example.jpg"
    )
    assert ranker_service._allowed_preview_url(
        "https://cdn.pixabay.com/video/example.jpg"
    )
    assert ranker_service._allowed_preview_url(
        "https://storage.coverr.co/t/example"
    )
    assert not ranker_service._allowed_preview_url("http://images.pexels.com/example.jpg")
    assert not ranker_service._allowed_preview_url("https://example.com/image.jpg")
    assert not ranker_service._allowed_preview_url(
        "https://pexels.com.evil.example/image.jpg"
    )


def test_ranker_service_aggregates_two_strongest_previews():
    assert ranker_service._aggregate([]) == -1.0
    assert ranker_service._aggregate([0.1]) == 0.1
    assert ranker_service._aggregate([0.1, 0.9, 0.7]) == 0.8
