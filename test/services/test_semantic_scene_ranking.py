from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.models.schema import MaterialInfo, VideoAspect, VideoParams
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


def test_build_negative_queries_targets_ambiguous_phone_and_writing_scenes():
    phone_negatives = semantic_ranker.build_negative_queries(
        "putting smartphone face down do not disturb desk"
    )
    assert "person holding a smartphone and looking at it" in phone_negatives
    assert "person talking on a smartphone" in phone_negatives
    assert "digital alarm clock on a desk" in phone_negatives
    assert "digital timer display" in phone_negatives

    writing_negatives = semantic_ranker.build_negative_queries(
        "hand writing next task in notebook to do list"
    )
    assert "person checking smartphone" in writing_negatives
    assert "person only typing on a keyboard" in writing_negatives


def test_reference_preview_urls_prioritize_recent_selected_clips():
    first = _item("first", "https://images.pexels.com/first.jpg")
    second = _item("second", "https://images.pexels.com/second.jpg")

    assert semantic_ranker.reference_preview_urls([first, second]) == [
        "https://images.pexels.com/second.jpg",
        "https://images.pexels.com/first.jpg",
    ]


def test_rank_materials_reorders_candidates_and_attaches_component_scores():
    first = _item("first", "https://images.pexels.com/first.jpg")
    second = _item("second", "https://images.pexels.com/second.jpg")
    reference = _item("reference", "https://images.pexels.com/reference.jpg")

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model": "ViT-B-32/test",
        "ranked": [
            {
                "id": "pexels:second",
                "score": 0.51,
                "positive_score": 0.81,
                "negative_score": 0.44,
                "diversity_similarity": 0.91,
            },
            {
                "id": "pexels:first",
                "score": 0.32,
                "positive_score": 0.42,
                "negative_score": 0.15,
                "diversity_similarity": 0.40,
            },
        ],
    }
    session = Mock()
    session.post.return_value = response

    with patch("app.services.semantic_ranker._local_session", return_value=session):
        ranked = semantic_ranker.rank_materials(
            "hand writing in notebook",
            [first, second],
            enabled=True,
            reference_items=[reference],
        )

    assert [item.source_info["asset_id"] for item in ranked] == ["second", "first"]
    assert ranked[0].source_info["semantic_score"] == 0.51
    assert ranked[0].source_info["semantic_positive_score"] == 0.81
    assert ranked[0].source_info["semantic_negative_score"] == 0.44
    assert ranked[0].source_info["semantic_diversity_similarity"] == 0.91
    assert ranked[0].source_info["semantic_rank"] == 1
    assert ranked[0].source_info["semantic_ranker_model"] == "ViT-B-32/test"
    assert ranked[0].source_info["semantic_reference_count"] == 1

    payload = session.post.call_args.kwargs["json"]
    assert "person checking smartphone" in payload["negative_queries"]
    assert payload["reference_preview_urls"] == [
        "https://images.pexels.com/reference.jpg"
    ]
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
            video_aspect=VideoAspect.portrait,
            audio_duration=3,
            max_clip_duration=3,
            material_directory="/tmp",
            semantic_scene_ranking=True,
        )

    assert Path(paths[0]).name == "second.mp4"
    rerank.assert_called_once()
    assert rerank.call_args.kwargs["reference_items"] == []


def test_strict_scene_reranks_repeated_query_with_selected_visual_context():
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
            side_effect=lambda query, items, **kwargs: list(items),
        ) as rerank,
    ):
        paths = strict_scene.download_videos_by_scene_queries(
            task_id="semantic-context-test",
            search_terms=["focused work"],
            search_videos=search_videos,
            save_video=save_video,
            source_record=lambda item, path: {"asset_id": item.source_info["asset_id"]},
            persist_sources=lambda task_id, sources: None,
            redact_error=lambda error, secret: str(error).replace(secret, "***"),
            video_aspect=VideoAspect.portrait,
            audio_duration=6,
            max_clip_duration=3,
            material_directory="/tmp",
            semantic_scene_ranking=True,
        )

    assert [Path(path).name for path in paths] == ["first.mp4", "second.mp4"]
    assert rerank.call_count == 2
    assert rerank.call_args_list[0].kwargs["reference_items"] == []
    assert rerank.call_args_list[1].kwargs["reference_items"] == [first]


def test_semantic_bridge_searches_transition_before_repeating_primary_intent():
    first = _item("first", "https://images.pexels.com/first.jpg")
    transition = _item("transition", "https://images.pexels.com/transition.jpg")
    second = _item("second", "https://images.pexels.com/second.jpg")
    bridge_query = strict_scene._build_bridge_query("morning coffee", "focused work")
    captured_plans = []

    def search_videos(search_term, minimum_duration, video_aspect):
        del minimum_duration, video_aspect
        if search_term == "morning coffee":
            first.source_info["search_term"] = search_term
            return [first]
        if search_term == "focused work":
            second.source_info["search_term"] = search_term
            return [second]
        if search_term == bridge_query:
            transition.source_info["search_term"] = search_term
            return [transition]
        return []

    def save_video(video_url, save_dir):
        del save_dir
        return str(Path("/tmp") / Path(video_url).name)

    def capture_plan(task_id, **updates):
        del task_id
        if "scene_plan" in updates:
            captured_plans.append([dict(scene) for scene in updates["scene_plan"]])
        return True

    with (
        patch(
            "app.services.strict_scene.task_artifacts.patch_script_data",
            side_effect=capture_plan,
        ),
        patch(
            "app.services.strict_scene.semantic_ranker.rank_materials",
            side_effect=lambda query, items, **kwargs: list(items),
        ) as rerank,
    ):
        paths = strict_scene.download_videos_by_scene_queries(
            task_id="semantic-bridge-test",
            search_terms=["morning coffee", "focused work"],
            search_videos=search_videos,
            save_video=save_video,
            source_record=lambda item, path: {
                "asset_id": item.source_info["asset_id"]
            },
            persist_sources=lambda task_id, sources: None,
            redact_error=lambda error, secret: str(error).replace(secret, "***"),
            video_aspect=VideoAspect.portrait,
            audio_duration=7,
            max_clip_duration=3,
            material_directory="/tmp",
            semantic_scene_ranking=True,
        )

    assert [Path(path).name for path in paths] == [
        "first.mp4",
        "transition.mp4",
        "second.mp4",
    ]
    assert any(call.args[0] == bridge_query for call in rerank.call_args_list)
    bridge_scene = captured_plans[-1][1]
    assert bridge_scene["semantic_bridge"] is True
    assert bridge_scene["bridge_selected"] is True
    assert bridge_scene["selected_query"] == bridge_query


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
    assert ranker_service._allowed_preview_url("https://storage.coverr.co/t/example")
    assert not ranker_service._allowed_preview_url("http://images.pexels.com/example.jpg")
    assert not ranker_service._allowed_preview_url("https://example.com/image.jpg")
    assert not ranker_service._allowed_preview_url(
        "https://pexels.com.evil.example/image.jpg"
    )


def test_ranker_service_aggregates_two_strongest_previews():
    assert ranker_service._aggregate([]) == -1.0
    assert ranker_service._aggregate([0.1]) == 0.1
    assert ranker_service._aggregate([0.1, 0.9, 0.7]) == 0.8


def test_ranker_service_combines_negative_and_diversity_penalties():
    score = ranker_service._final_score(
        0.40,
        0.30,
        0.90,
        negative_weight=0.45,
        diversity_weight=0.35,
        diversity_threshold=0.80,
    )
    assert score == pytest.approx(0.23)


def test_ranker_candidate_metrics_penalize_negative_visual_concept():
    metrics = ranker_service._candidate_metrics(
        image_features=[[0.8, 0.6]],
        positive_feature=[1.0, 0.0],
        negative_features=[[0.0, 1.0]],
        reference_features=[],
    )
    assert metrics["positive_score"] == pytest.approx(0.8)
    assert metrics["negative_score"] == pytest.approx(0.6)
    assert metrics["score"] < metrics["positive_score"]



def test_monotonic_narration_alignment_reaches_final_term_before_tiny_tail():
    plan = strict_scene.build_scene_plan(
        ["first", "middle", "closing"],
        audio_duration=9.1,
        max_clip_duration=3,
    )
    scores = [
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.0, 0.0, 0.95],
    ]
    alignment = strict_scene._monotonic_alignment(scores, plan, 3)
    assert alignment == [0, 1, 2, 2]
    assert plan[-1]["duration"] == pytest.approx(0.1)


def test_apply_narration_alignment_uses_real_subtitle_text(tmp_path):
    subtitle_path = tmp_path / "narration.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,900\nphone in bed\n\n"
        "2\n00:00:03,000 --> 00:00:05,900\nwrite the task down\n\n",
        encoding="utf-8",
    )
    plan = strict_scene.build_scene_plan(
        ["checking smartphone", "writing notebook"],
        audio_duration=6,
        max_clip_duration=3,
    )
    with patch(
        "app.services.strict_scene.semantic_ranker.align_scene_terms",
        return_value=[[0.9, 0.1], [0.1, 0.9]],
    ):
        applied = strict_scene._apply_narration_alignment(
            plan,
            ["checking smartphone", "writing notebook"],
            str(subtitle_path),
            enabled=True,
        )
    assert applied is True
    assert [scene["query_index"] for scene in plan] == [0, 1]
    assert plan[0]["narration_text"] == "phone in bed"
    assert plan[1]["narration_text"] == "write the task down"


def test_alignment_service_returns_text_similarity_matrix(monkeypatch):
    fake_features = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    monkeypatch.setattr(ranker_service, "_encode_texts", lambda texts: fake_features)
    monkeypatch.setattr(
        ranker_service,
        "_load_model",
        lambda: {"model_name": "test", "pretrained": "unit", "device": "cpu"},
    )
    result = ranker_service._align(
        ranker_service.AlignRequest(
            scenes=["scene one", "scene two"],
            terms=["term one", "term two"],
        )
    )
    assert result["scores"] == [[1.0, 0.0], [0.0, 1.0]]
