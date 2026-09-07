"""Deterministic stock-footage assignment for strict scene matching.

This module is intentionally independent from any LLM. Ordered search terms supplied by
MoneyPrinterTurbo (or entered manually by the user) are treated as visual scene intents.
The downloader keeps those intents in narrative order, prefers a unique provider asset
for every scene, and only reuses an asset when the provider has no usable alternative.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, List

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.services import gpt_visual_review, semantic_ranker, subtitle, task_artifacts

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
    _mark_semantic_bridges(plan, terms)
    return plan



def _mark_semantic_bridges(
    plan: list[dict[str, Any]],
    terms: list[str],
) -> None:
    for scene in plan:
        for key in ("bridge_to_query_index", "bridge_to_query", "bridge_query"):
            scene.pop(key, None)

    run_start = 0
    while run_start < len(plan):
        run_query_index = int(plan[run_start]["query_index"])
        run_end = run_start + 1
        while (
            run_end < len(plan)
            and int(plan[run_end]["query_index"]) == run_query_index
        ):
            run_end += 1

        if run_end - run_start > 1 and run_end < len(plan):
            next_query_index = int(plan[run_end]["query_index"])
            if next_query_index > run_query_index:
                bridge_scene = plan[run_end - 1]
                bridge_scene["bridge_to_query_index"] = next_query_index
                bridge_scene["bridge_to_query"] = terms[next_query_index]
                bridge_scene["bridge_query"] = _build_bridge_query(
                    terms[run_query_index],
                    terms[next_query_index],
                )
        run_start = run_end


def _build_bridge_query(
    current_query: str,
    next_query: str,
    *,
    max_chars: int = 180,
) -> str:
    """Build a bounded provider/CLIP query that represents a narrative transition."""

    current = " ".join(str(current_query or "").split())
    following = " ".join(str(next_query or "").split())
    if not current:
        return following[:max_chars].strip()
    if not following:
        return current[:max_chars].strip()

    combined = f"{current}; {following}"
    if len(combined) <= max_chars:
        return combined

    # Preserve useful text from both sides instead of truncating away the next
    # narrative intent when user-entered keywords are unusually verbose.
    usable = max(8, max_chars - 2)
    current_budget = usable // 2
    following_budget = usable - current_budget

    def trim(value: str, budget: int) -> str:
        if len(value) <= budget:
            return value
        shortened = value[:budget].rsplit(" ", 1)[0].strip()
        return shortened or value[:budget].strip()

    return f"{trim(current, current_budget)}; {trim(following, following_budget)}"



_SRT_TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+),(?P<ms>\d+)")
_MIN_MEANINGFUL_TAIL_SECONDS = 1.0
_POSITION_PRIOR_WEIGHT = 0.08


def _srt_time_seconds(value: str) -> float | None:
    match = _SRT_TIME_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    parts = {key: int(raw) for key, raw in match.groupdict().items()}
    return (
        parts["h"] * 3600
        + parts["m"] * 60
        + parts["s"]
        + parts["ms"] / 1000.0
    )


def _subtitle_cues(subtitle_path: str) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    if not subtitle_path:
        return cues
    for _, times, text in subtitle.file_to_subtitles(subtitle_path):
        try:
            start_raw, end_raw = [part.strip() for part in times.split(" --> ", 1)]
        except ValueError:
            continue
        start = _srt_time_seconds(start_raw)
        end = _srt_time_seconds(end_raw)
        narration = " ".join(str(text or "").split())
        if start is None or end is None or end <= start or not narration:
            continue
        cues.append((start, end, narration))
    return cues


def _attach_narration_text(
    scene_plan: list[dict[str, Any]],
    subtitle_path: str,
) -> list[str]:
    cues = _subtitle_cues(subtitle_path)
    scene_texts: list[str] = []
    for scene in scene_plan:
        start = float(scene["start"])
        end = float(scene["end"])
        overlapping = [
            text
            for cue_start, cue_end, text in cues
            if cue_end > start and cue_start < end
        ]
        narration = " ".join(overlapping)
        scene["narration_text"] = narration
        scene_texts.append(narration)
    return scene_texts


def _monotonic_alignment(
    scores: list[list[float]],
    scene_plan: list[dict[str, Any]],
    term_count: int,
) -> list[int] | None:
    """Choose a nondecreasing scene->term path with real closing-term exposure."""
    scene_count = len(scene_plan)
    if (
        scene_count == 0
        or term_count <= 0
        or len(scores) != scene_count
        or scene_count < term_count
    ):
        return None
    if any(len(row) != term_count for row in scores):
        return None

    meaningful_end = scene_count - 1
    if (
        float(scene_plan[-1].get("duration") or 0) < _MIN_MEANINGFUL_TAIL_SECONDS
        and scene_count > 1
    ):
        meaningful_end -= 1
    active_count = meaningful_end + 1
    if active_count < term_count:
        return None

    neg_inf = float("-inf")
    dp = [[neg_inf] * term_count for _ in range(active_count)]
    prev = [[-1] * term_count for _ in range(active_count)]

    def weighted_score(scene_index: int, term_index: int) -> float:
        reference_duration = max(float(scene_plan[0].get("duration") or 1), 0.001)
        duration_weight = max(
            0.15,
            float(scene_plan[scene_index].get("duration") or 0) / reference_duration,
        )
        if active_count <= 1 or term_count <= 1:
            position_prior = 1.0
        else:
            scene_position = scene_index / (active_count - 1)
            term_position = term_index / (term_count - 1)
            position_prior = 1.0 - abs(scene_position - term_position)
        return (scores[scene_index][term_index] * duration_weight) + (
            _POSITION_PRIOR_WEIGHT * position_prior
        )

    dp[0][0] = weighted_score(0, 0)
    for scene_index in range(1, active_count):
        for term_index in range(term_count):
            for previous_term in (term_index, term_index - 1):
                if previous_term < 0 or dp[scene_index - 1][previous_term] == neg_inf:
                    continue
                candidate = dp[scene_index - 1][previous_term] + weighted_score(
                    scene_index, term_index
                )
                if candidate > dp[scene_index][term_index]:
                    dp[scene_index][term_index] = candidate
                    prev[scene_index][term_index] = previous_term

    if dp[meaningful_end][term_count - 1] == neg_inf:
        return None

    alignment = [0] * active_count
    current = term_count - 1
    for scene_index in range(meaningful_end, -1, -1):
        alignment[scene_index] = current
        if scene_index > 0:
            current = prev[scene_index][current]
            if current < 0:
                return None

    alignment.extend([term_count - 1] * (scene_count - active_count))
    return alignment


def _apply_narration_alignment(
    scene_plan: list[dict[str, Any]],
    terms: list[str],
    subtitle_path: str,
    *,
    enabled: bool,
) -> bool:
    scene_texts = _attach_narration_text(scene_plan, subtitle_path)
    similarity = semantic_ranker.align_scene_terms(
        scene_texts,
        terms,
        enabled=enabled,
    )
    if similarity is None:
        return False
    alignment = _monotonic_alignment(similarity, scene_plan, len(terms))
    if alignment is None:
        return False

    for scene, term_index, row in zip(scene_plan, alignment, similarity):
        scene["query_index"] = term_index
        scene["query"] = terms[term_index]
        scene["narration_aligned"] = True
        scene["narration_alignment_score"] = round(float(row[term_index]), 6)
    return True


_SCENE_QUERY_RRF_K = 60.0
_SCENE_SPECIFIC_WEIGHT = 1.15
_ANCHOR_QUERY_WEIGHT = 1.0
_BRIDGE_NEIGHBOR_WEIGHT = 0.85
_PROVIDER_QUERY_MAX_WORDS = 18
_PROVIDER_QUERY_MAX_CHARS = 160
_SEMANTIC_QUERY_MAX_WORDS = 30
_PROVIDER_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_PROVIDER_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "can", "could", "did", "do", "does", "for", "from", "had",
        "has", "have", "he", "her", "hers", "him", "his", "how", "i", "if",
        "in", "into", "is", "it", "its", "may", "might", "of", "on", "or",
        "our", "ours", "she", "so", "some", "something", "that", "the",
        "their", "theirs", "them", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "up", "us", "was", "we", "were",
        "what", "when", "where", "which", "while", "who", "why", "will",
        "with", "would", "you", "your", "yours",
    }
)


def _provider_safe_text(value: str) -> str:
    """Remove provider-hostile negation while preserving a positive visual cue."""
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    text = re.sub(
        r"\bdo\s+not\s+disturb\b",
        "phone face down silent notifications",
        text,
        flags=re.IGNORECASE,
    )
    # Dropping the whole negated phone phrase is intentional.  Keeping only the
    # noun would turn "no phone" into a provider search for "phone".
    text = re.sub(
        r"\bwithout\s+(?:a\s+|the\s+)?phone\b|\bno\s+phone\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split())


def _provider_query_words(value: str) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for raw in _PROVIDER_TOKEN_RE.findall(_provider_safe_text(value).lower()):
        word = raw.strip("'-")
        if not word or word in _PROVIDER_STOPWORDS or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def _bounded_query(words: list[str]) -> str:
    selected: list[str] = []
    for word in words:
        if len(selected) >= _PROVIDER_QUERY_MAX_WORDS:
            break
        candidate = " ".join(selected + [word])
        if len(candidate) > _PROVIDER_QUERY_MAX_CHARS:
            break
        selected.append(word)
    return " ".join(selected)


def _build_scene_specific_query(narration: str, anchor_query: str) -> str:
    """Create a short stock-provider query from visual anchor + timed narration.

    The manually supplied anchor stays first because it is already written as a
    visual intent.  Timed narration contributes only new content words, giving
    each 3-second scene a more specific search without requiring another LLM.
    """
    anchor_words = _provider_query_words(anchor_query)
    narration_words = _provider_query_words(narration)
    merged = list(anchor_words)
    seen = set(anchor_words)
    for word in narration_words:
        if word in seen:
            continue
        seen.add(word)
        merged.append(word)
    return _bounded_query(merged)


def _build_semantic_scene_query(narration: str, anchor_query: str) -> str:
    """Build a bounded natural-language CLIP query for the final preview rerank."""
    pieces = [
        " ".join(str(anchor_query or "").split()),
        " ".join(str(narration or "").split()),
    ]
    combined = ". ".join(piece for piece in pieces if piece)
    words = combined.split()
    if len(words) > _SEMANTIC_QUERY_MAX_WORDS:
        combined = " ".join(words[:_SEMANTIC_QUERY_MAX_WORDS])
    return combined.strip(" ,;.-")


def _query_specs_for_scene(
    narration: str,
    anchor_query: str,
) -> list[tuple[str, float, str]]:
    """Return scene-specific and anchor searches, deduplicated in priority order."""
    specific = _build_scene_specific_query(narration, anchor_query)
    specs = [
        (specific, _SCENE_SPECIFIC_WEIGHT, "scene_specific"),
        (_provider_safe_text(anchor_query), _ANCHOR_QUERY_WEIGHT, "anchor"),
    ]
    output: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    for query, weight, kind in specs:
        normalized = " ".join(str(query or "").split()).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        output.append((normalized, weight, kind))
    return output


def _fuse_provider_rankings(
    ranked_results: list[tuple[str, float, List[MaterialInfo]]],
) -> List[MaterialInfo]:
    """Fuse multiple provider result lists with weighted reciprocal-rank fusion.

    Provider APIs expose useful ordering but not comparable relevance scores.
    RRF combines those ranks without score normalization and rewards an asset that
    appears under more than one scene phrasing before OpenCLIP performs the final
    visual rerank.
    """
    scores: dict[str, float] = {}
    hits: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    identity_to_item: dict[str, MaterialInfo] = {}
    encounter = 0

    for _, weight, items in ranked_results:
        for rank, item in enumerate(items, start=1):
            identity = _material_identity(item)
            if identity not in identity_to_item:
                identity_to_item[identity] = item
                first_seen[identity] = encounter
                encounter += 1
            scores[identity] = scores.get(identity, 0.0) + (
                float(weight) / (_SCENE_QUERY_RRF_K + rank)
            )
            hits[identity] = hits.get(identity, 0) + 1

    identities = sorted(
        identity_to_item,
        key=lambda identity: (-scores[identity], first_seen[identity]),
    )
    fused: list[MaterialInfo] = []
    for identity in identities:
        item = identity_to_item[identity]
        source = dict(item.source_info) if isinstance(item.source_info, dict) else {}
        source["provider_fusion_score"] = round(scores[identity], 8)
        source["provider_query_hits"] = hits[identity]
        item.source_info = source
        fused.append(item)
    return fused


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
    narration_subtitle_path: str = "",
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

    narration_aligned = _apply_narration_alignment(
        scene_plan,
        terms,
        narration_subtitle_path,
        enabled=semantic_scene_ranking,
    )
    if semantic_scene_ranking:
        _mark_semantic_bridges(scene_plan, terms)
    logger.info(
        "strict scene narration alignment: "
        f"enabled={semantic_scene_ranking}, applied={narration_aligned}"
    )

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

    def fused_raw_candidates(
        specs: list[tuple[str, float, str]],
    ) -> List[MaterialInfo]:
        ranked_results = [
            (query, weight, raw_candidates_for(query))
            for query, weight, _ in specs
        ]
        return _fuse_provider_rankings(ranked_results)

    def record_scene_queries(
        scene: dict[str, Any],
        specs: list[tuple[str, float, str]],
    ) -> None:
        existing = scene.setdefault("search_queries", [])
        known = {str(value.get("query") or "").lower() for value in existing if isinstance(value, dict)}
        for query, weight, kind in specs:
            if query.lower() in known:
                continue
            existing.append(
                {"query": query, "weight": round(float(weight), 3), "kind": kind}
            )
            known.add(query.lower())

    def candidates_for_scene(
        scene: dict[str, Any],
        anchor_query: str,
    ) -> List[MaterialInfo]:
        if not semantic_scene_ranking:
            return raw_candidates_for(anchor_query)

        narration = str(scene.get("narration_text") or "")
        specs = _query_specs_for_scene(narration, anchor_query)
        record_scene_queries(scene, specs)
        semantic_query = _build_semantic_scene_query(narration, anchor_query)
        scene["semantic_query"] = semantic_query
        return semantic_ranker.rank_materials(
            semantic_query,
            fused_raw_candidates(specs),
            enabled=True,
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
        bridge_ranked: list[MaterialInfo] = []
        bridge_selected = False
        review_ranked: list[MaterialInfo] = []

        # A semantic bridge is intentionally opt-in with Semantic Scene Ranking.
        # It searches a combined current->next intent plus both original pools,
        # then lets CLIP and the existing reference-image penalty pick a transition
        # shot. If the combined query is sparse or the ranker is offline, the
        # ordinary strict fallback path below remains intact.
        if semantic_scene_ranking:
            bridge_to_index = scene.get("bridge_to_query_index")
            bridge_query = str(scene.get("bridge_query") or "").strip()
            if (
                isinstance(bridge_to_index, int)
                and 0 <= bridge_to_index < len(terms)
                and bridge_query
            ):
                narration = str(scene.get("narration_text") or "")
                bridge_specs = _query_specs_for_scene(narration, bridge_query)
                existing_bridge_queries = {query.lower() for query, _, _ in bridge_specs}
                for query, weight, kind in (
                    (terms[primary_index], _BRIDGE_NEIGHBOR_WEIGHT, "bridge_current"),
                    (terms[bridge_to_index], _BRIDGE_NEIGHBOR_WEIGHT, "bridge_next"),
                ):
                    safe_query = _provider_safe_text(query)
                    if safe_query and safe_query.lower() not in existing_bridge_queries:
                        bridge_specs.append((safe_query, weight, kind))
                        existing_bridge_queries.add(safe_query.lower())
                record_scene_queries(scene, bridge_specs)
                bridge_raw = fused_raw_candidates(bridge_specs)
                semantic_query = _build_semantic_scene_query(narration, bridge_query)
                bridge_ranked = semantic_ranker.rank_materials(
                    semantic_query,
                    bridge_raw,
                    enabled=True,
                    reference_items=list(selected_materials),
                )
                review_ranked = list(bridge_ranked)
                scene["semantic_bridge"] = True
                scene["semantic_query"] = semantic_query
                for item in bridge_ranked:
                    result = try_candidate(item, allow_reuse=False)
                    if result is None:
                        continue
                    selected = result
                    selected_item = item
                    source = (
                        item.source_info
                        if isinstance(item.source_info, dict)
                        else {}
                    )
                    selected_query = str(
                        source.get("search_term") or bridge_query
                    )
                    bridge_selected = True
                    break

        # First pass: only unused source assets.  Semantic mode searches a
        # scene-specific rewrite plus the original anchor and fuses both provider
        # rankings before OpenCLIP.  Strict-only mode remains byte-for-byte
        # equivalent in intent: a single anchor query at a time.
        if selected is None:
            for query_index in query_indexes:
                query = terms[query_index]
                ranked_candidates = candidates_for_scene(scene, query)
                review_ranked = list(ranked_candidates)
                for item in ranked_candidates:
                    result = try_candidate(item, allow_reuse=False)
                    if result is None:
                        continue
                    selected = result
                    selected_item = item
                    source = item.source_info if isinstance(item.source_info, dict) else {}
                    selected_query = str(source.get("search_term") or query)
                    break
                if selected is not None:
                    break

        # If every unused bridge candidate failed to download, allow bridge reuse
        # before falling all the way back to the ordinary nearest-query pool.
        if selected is None and bridge_ranked:
            bridge_query = str(scene.get("bridge_query") or "").strip()
            for item in bridge_ranked:
                result = try_candidate(item, allow_reuse=True)
                if result is None:
                    continue
                selected = result
                selected_item = item
                source = (
                    item.source_info
                    if isinstance(item.source_info, dict)
                    else {}
                )
                selected_query = str(source.get("search_term") or bridge_query)
                bridge_selected = True
                break

        # Last resort: allow a source to repeat rather than leaving the timeline short.
        if selected is None:
            for query_index in query_indexes:
                query = terms[query_index]
                ranked_candidates = candidates_for_scene(scene, query)
                review_ranked = list(ranked_candidates)
                for item in ranked_candidates:
                    result = try_candidate(item, allow_reuse=True)
                    if result is None:
                        continue
                    selected = result
                    selected_item = item
                    source = item.source_info if isinstance(item.source_info, dict) else {}
                    selected_query = str(source.get("search_term") or query)
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
                "bridge_selected": bool(bridge_selected),
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
        gpt_visual_review.record_scene_candidates(
            task_id=task_id,
            scene=scene,
            ranked_items=review_ranked or [selected_item],
            selected_item=selected_item,
            selected_path=saved_path,
            selected_query=selected_query,
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
