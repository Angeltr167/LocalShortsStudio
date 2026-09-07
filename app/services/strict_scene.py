"""Deterministic stock-footage assignment for strict scene matching.

This module is intentionally independent from any LLM. Ordered search terms supplied by
MoneyPrinterTurbo (or entered manually by the user) are treated as visual scene intents.
The downloader keeps those intents in narrative order, prefers a unique provider asset
for every scene, and only reuses an asset when the provider has no usable alternative.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, List

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.services import semantic_ranker, task_artifacts

SearchVideos = Callable[..., List[MaterialInfo]]
SaveVideo = Callable[..., str]
SourceRecord = Callable[[MaterialInfo, str], dict[str, Any]]
PersistSources = Callable[[str, list[dict[str, Any]]], None]
RedactError = Callable[..., str]


def normalize_scene_search_terms(search_terms: List[str]) -> list[str]:
    """Return non-empty stock-footage queries while preserving user order."""
    return [str(term).strip() for term in search_terms if str(term or "").strip()]


def build_scene_plan(
    search_terms: List[str],
    audio_duration: float,
    max_clip_duration: int,
) -> list[dict[str, Any]]:
    """Build clip-sized timeline slots from the real narration duration.

    Search terms are distributed monotonically over the timeline. If there are fewer
    queries than scene slots, each query repeats only in adjacent slots instead of
    cycling back to an earlier topic later in the video.
    """
    terms = normalize_scene_search_terms(search_terms)
    if not terms:
        return []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError, OverflowError):
        return []
    if not math.isfinite(required_duration) or required_duration <= 0:
        return []

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError):
        return []
    if clip_duration <= 0:
        return []

    scene_count = max(1, math.ceil(required_duration / clip_duration))
    plan: list[dict[str, Any]] = []
    for scene_index in range(scene_count):
        query_index = min(
            len(terms) - 1,
            math.floor(scene_index * len(terms) / scene_count),
        )
        start = scene_index * clip_duration
        end = min(required_duration, start + clip_duration)
        plan.append(
            {
                "scene": scene_index + 1,
                "query_index": query_index,
                "query": terms[query_index],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "status": "pending",
            }
        )
    return plan


def _fallback_query_indexes(primary_index: int, query_count: int) -> list[int]:
    """Return the primary query followed by the nearest narrative neighbors."""
    order = [primary_index]
    for distance in range(1, query_count):
        following = primary_index + distance
        previous = primary_index - distance
        if following < query_count:
            order.append(following)
        if previous >= 0:
            order.append(previous)
    return order


def _material_identity(item: MaterialInfo) -> str:
    """Prefer provider asset ids so alternate renditions still deduplicate."""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    provider = str(item.provider or source.get("provider") or "unknown")
    asset_id = source.get("asset_id")
    if asset_id not in (None, ""):
        return f"{provider}:asset:{asset_id}"
    return f"{provider}:url:{item.url}"


def download_videos_by_scene_queries(
    *,
    task_id: str,
    search_terms: List[str],
    search_videos: SearchVideos,
    save_video: SaveVideo,
    source_record: SourceRecord,
    persist_sources: PersistSources,
    redact_error: RedactError,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    semantic_scene_ranking: bool = False,
) -> List[str]:
    """Download one ordered stock clip per scene slot.

    The primary query is always exhausted before neighboring queries are considered.
    Duplicate provider assets are skipped while unused candidates exist. Reuse is a
    final fallback so a sparse provider result set does not unnecessarily fail a task.

    When semantic ranking is enabled, provider search results remain cached but their
    semantic ordering is recomputed for each timeline scene. The ranker receives previews
    from clips already selected so visually repetitive alternatives can be penalized.
    """
    terms = normalize_scene_search_terms(search_terms)
    scene_plan = build_scene_plan(
        search_terms=terms,
        audio_duration=audio_duration,
        max_clip_duration=max_clip_duration,
    )
    if not scene_plan:
        logger.warning("strict scene matching could not build a scene plan")
        return []

    task_artifacts.patch_script_data(task_id, scene_plan=scene_plan)
    logger.info(
        "downloading videos with strict scene matching: "
        f"scenes={len(scene_plan)}, queries={len(terms)}"
    )

    raw_candidate_cache: dict[str, List[MaterialInfo]] = {}
    used_material_ids: set[str] = set()
    failed_material_ids: set[str] = set()
    selected_materials: list[MaterialInfo] = []
    video_paths: list[str] = []
    material_sources: list[dict[str, Any]] = []

    def raw_candidates_for(query: str) -> List[MaterialInfo]:
        if query not in raw_candidate_cache:
            raw_candidate_cache[query] = list(
                search_videos(
                    search_term=query,
                    minimum_duration=max_clip_duration,
                    video_aspect=video_aspect,
                )
            )
            logger.info(
                f"found {len(raw_candidate_cache[query])} strict candidates for {query!r}"
            )
        return raw_candidate_cache[query]

    def candidates_for(query: str) -> List[MaterialInfo]:
        return semantic_ranker.rank_materials(
            query,
            raw_candidates_for(query),
            enabled=semantic_scene_ranking,
            reference_items=list(selected_materials),
        )

    def try_candidate(
        item: MaterialInfo,
        *,
        allow_reuse: bool,
    ) -> tuple[str, bool] | None:
        identity = _material_identity(item)
        if identity in failed_material_ids:
            return None
        already_used = identity in used_material_ids
        if already_used and not allow_reuse:
            return None

        try:
            saved_path = save_video(
                video_url=item.url,
                save_dir=material_directory,
            )
        except Exception as exc:
            failed_material_ids.add(identity)
            logger.error(
                "failed to download strict scene candidate: "
                f"provider={item.provider}, error={type(exc).__name__}, "
                f"detail={redact_error(exc, item.url)}"
            )
            return None
        if not saved_path:
            failed_material_ids.add(identity)
            return None

        used_material_ids.add(identity)
        return saved_path, already_used

    for scene in scene_plan:
        primary_index = int(scene["query_index"])
        query_indexes = _fallback_query_indexes(primary_index, len(terms))
        selected: tuple[str, bool] | None = None
        selected_item: MaterialInfo | None = None
        selected_query = ""

        # First pass: only unused source assets.
        for query_index in query_indexes:
            query = terms[query_index]
            for item in candidates_for(query):
                result = try_candidate(item, allow_reuse=False)
                if result is None:
                    continue
                selected = result
                selected_item = item
                selected_query = query
                break
            if selected is not None:
                break

        # Last resort: allow a source to repeat rather than leaving the timeline short.
        if selected is None:
            for query_index in query_indexes:
                query = terms[query_index]
                for item in candidates_for(query):
                    result = try_candidate(item, allow_reuse=True)
                    if result is None:
                        continue
                    selected = result
                    selected_item = item
                    selected_query = query
                    break
                if selected is not None:
                    break

        if selected is None or selected_item is None:
            scene["status"] = "unfilled"
            logger.warning(f"strict scene {scene['scene']} has no downloadable candidate")
            continue

        saved_path, reused = selected
        video_paths.append(saved_path)
        selected_materials.append(selected_item)
        source = (
            selected_item.source_info
            if isinstance(selected_item.source_info, dict)
            else {}
        )
        scene.update(
            {
                "status": "selected",
                "selected_query": selected_query,
                "provider": selected_item.provider,
                "asset_id": source.get("asset_id"),
                "local_file": Path(saved_path).name,
                "reused": bool(reused),
                "semantic_score": source.get("semantic_score"),
                "semantic_positive_score": source.get("semantic_positive_score"),
                "semantic_negative_score": source.get("semantic_negative_score"),
                "semantic_diversity_similarity": source.get(
                    "semantic_diversity_similarity"
                ),
                "semantic_rank": source.get("semantic_rank"),
                "semantic_ranker_model": source.get("semantic_ranker_model"),
                "semantic_reference_count": source.get("semantic_reference_count"),
            }
        )
        if reused:
            logger.warning(
                "strict scene matching reused a source as last resort: "
                f"scene={scene['scene']}, query={selected_query!r}"
            )

        try:
            record = source_record(selected_item, saved_path)
            record["scene_index"] = scene["scene"]
            record["scene_query"] = scene["query"]
            record["selected_query"] = selected_query
            record["reused"] = bool(reused)
            material_sources.append(record)
        except Exception as exc:
            logger.warning(
                "failed to prepare strict scene source record: "
                f"provider={selected_item.provider}, error={type(exc).__name__}, "
                f"detail={exc}"
            )

    task_artifacts.patch_script_data(task_id, scene_plan=scene_plan)
    persist_sources(task_id, material_sources)
    logger.success(
        "downloaded strict scene materials: "
        f"selected={len(video_paths)}, scenes={len(scene_plan)}, "
        f"unique_sources={len(used_material_ids)}"
    )
    return video_paths
