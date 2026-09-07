"""Optional local semantic reranking for stock-video candidates.

The MoneyPrinterTurbo process never imports OpenCLIP directly. Instead it sends
provider preview-image URLs to a small local service (default 127.0.0.1:4124).
If that service is unavailable or returns an invalid response, candidates are
returned in their original provider order so Strict Scene Matching remains a
safe fallback.
"""

from __future__ import annotations

import math
from typing import List

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo

DEFAULT_SEMANTIC_RANKER_BASE_URL = "http://127.0.0.1:4124"
MAX_CANDIDATES = 20
MAX_PREVIEWS_PER_CANDIDATE = 4
MAX_NEGATIVE_QUERIES = 8
MAX_REFERENCE_PREVIEWS = 12
_CONNECT_TIMEOUT_SECONDS = 2
_READ_TIMEOUT_SECONDS = 90

_NEGATIVE_QUERY_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("face down", "do not disturb", "silence notification", "silence notifications"),
        (
            "person holding a smartphone and looking at it",
            "person talking on a smartphone",
            "smartphone being actively used in hand",
            "digital alarm clock on a desk",
            "digital timer display",
            "desk clock close up",
        ),
    ),
    (
        ("no phone", "without phone"),
        (
            "person holding a smartphone",
            "person checking a phone",
            "phone screen close up",
        ),
    ),
    (
        ("deep work", "focused person", "focused worker", "productive person"),
        (
            "distracted person checking smartphone",
            "person multitasking between phone and computer",
        ),
    ),
    (
        ("writing", "notebook", "to do list", "checklist", "write down"),
        (
            "person checking smartphone",
            "phone screen close up",
            "person only typing on a keyboard",
        ),
    ),
    (
        ("multiple computer screens", "multiple screens", "multitasking"),
        (
            "single task focused worker at one laptop",
            "person writing calmly in a notebook",
        ),
    ),
    (
        ("checking email", "email"),
        (
            "person talking on a phone",
            "social media scrolling on smartphone",
        ),
    ),
    (
        ("notification", "notifications", "interrupting work"),
        (
            "smartphone lying unused face down",
            "focused work with no phone visible",
        ),
    ),
)


def _evenly_spaced(values: list[str], limit: int) -> list[str]:
    """Return up to ``limit`` values spread across the original ordering."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    if len(unique) <= limit:
        return unique
    if limit <= 1:
        return unique[:1]

    indexes = [round(i * (len(unique) - 1) / (limit - 1)) for i in range(limit)]
    return [unique[index] for index in indexes]


def preview_urls(item: MaterialInfo) -> list[str]:
    """Read transient provider preview URLs without persisting them to task artifacts."""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    raw = source.get("preview_images")
    values = raw if isinstance(raw, list) else []
    return _evenly_spaced(values, MAX_PREVIEWS_PER_CANDIDATE)


def build_negative_queries(query: str) -> list[str]:
    """Build deterministic anti-concepts for visually ambiguous stock searches.

    These rules do not call an LLM. They only add contrastive descriptions when
    the scene query contains a cue where CLIP commonly over-rewards a generic
    object (for example any phone instead of a phone lying face down).
    """
    normalized = " ".join(str(query or "").lower().split())
    if not normalized:
        return []

    negatives: list[str] = []
    seen: set[str] = set()
    for cues, descriptions in _NEGATIVE_QUERY_RULES:
        if not any(cue in normalized for cue in cues):
            continue
        for description in descriptions:
            if description in seen:
                continue
            seen.add(description)
            negatives.append(description)
            if len(negatives) >= MAX_NEGATIVE_QUERIES:
                return negatives
    return negatives


def reference_preview_urls(items: List[MaterialInfo]) -> list[str]:
    """Return a bounded set of previews from clips already chosen on the timeline."""
    values: list[str] = []
    seen: set[str] = set()
    # Recent selections matter most for perceived repetition, so walk backwards.
    for item in reversed(items):
        for value in _evenly_spaced(preview_urls(item), 2):
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
            if len(values) >= MAX_REFERENCE_PREVIEWS:
                return values
    return values


def _candidate_id(item: MaterialInfo, index: int) -> str:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    provider = str(item.provider or source.get("provider") or "unknown")
    asset_id = source.get("asset_id")
    return f"{provider}:{asset_id if asset_id not in (None, '') else index}"


def _base_url() -> str:
    configured = str(
        config.app.get("semantic_ranker_base_url", DEFAULT_SEMANTIC_RANKER_BASE_URL)
        or DEFAULT_SEMANTIC_RANKER_BASE_URL
    ).strip()
    return configured.rstrip("/")


def _local_session(base_url: str) -> requests.Session:
    """Bypass environment proxy settings for loopback ranker calls."""
    session = requests.Session()
    if base_url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
        session.trust_env = False
    return session


def rank_materials(
    query: str,
    items: List[MaterialInfo],
    *,
    enabled: bool,
    reference_items: List[MaterialInfo] | None = None,
) -> List[MaterialInfo]:
    """Rerank stock materials by contrastive semantics and visual diversity.

    The function is deliberately fail-open: any transport, schema, or scoring
    failure returns the provider ordering unchanged.
    """
    ordered = list(items)
    if not enabled or len(ordered) < 2:
        return ordered

    selected = ordered[:MAX_CANDIDATES]
    id_to_item: dict[str, MaterialInfo] = {}
    payload_candidates = []
    for index, item in enumerate(selected):
        candidate_id = _candidate_id(item, index)
        # Provider asset ids should be unique. If an unexpected duplicate exists,
        # retain both candidates by suffixing the list index.
        if candidate_id in id_to_item:
            candidate_id = f"{candidate_id}:{index}"
        id_to_item[candidate_id] = item
        payload_candidates.append(
            {
                "id": candidate_id,
                "preview_urls": preview_urls(item),
            }
        )

    if not any(candidate["preview_urls"] for candidate in payload_candidates):
        logger.info("semantic scene ranking skipped: candidates have no preview images")
        return ordered

    negatives = build_negative_queries(query)
    references = reference_preview_urls(reference_items or [])
    base_url = _base_url()
    payload = {
        "query": str(query or "").strip(),
        "negative_queries": negatives,
        "reference_preview_urls": references,
        "candidates": payload_candidates,
    }

    try:
        session = _local_session(base_url)
        try:
            response = session.post(
                f"{base_url}/rank",
                json=payload,
                timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
            )
        finally:
            session.close()
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        logger.warning(
            "semantic scene ranker unavailable, keep provider ordering: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return ordered

    ranked_rows = body.get("ranked") if isinstance(body, dict) else None
    if not isinstance(ranked_rows, list):
        logger.warning("semantic scene ranker returned an invalid response, keep provider ordering")
        return ordered

    ranked_items: list[MaterialInfo] = []
    used_ids: set[str] = set()
    for rank_index, row in enumerate(ranked_rows, start=1):
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("id") or "")
        item = id_to_item.get(candidate_id)
        if item is None or candidate_id in used_ids:
            continue

        source = dict(item.source_info) if isinstance(item.source_info, dict) else {}
        for response_key, source_key in (
            ("score", "semantic_score"),
            ("positive_score", "semantic_positive_score"),
            ("negative_score", "semantic_negative_score"),
            ("diversity_similarity", "semantic_diversity_similarity"),
        ):
            try:
                score = float(row.get(response_key))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(score):
                source[source_key] = round(score, 6)

        source["semantic_rank"] = rank_index
        model_name = body.get("model") if isinstance(body, dict) else None
        if model_name:
            source["semantic_ranker_model"] = str(model_name)
        if negatives:
            source["semantic_negative_queries"] = list(negatives)
        source["semantic_reference_count"] = len(references)
        item.source_info = source
        ranked_items.append(item)
        used_ids.add(candidate_id)

    if not ranked_items:
        logger.warning("semantic scene ranker produced no usable rows, keep provider ordering")
        return ordered

    # Preserve any candidates omitted by the service, followed by items beyond the
    # configured ranking cap. This keeps the downloader's fallback pool complete.
    ranked_object_ids = {id(item) for item in ranked_items}
    ranked_items.extend(item for item in selected if id(item) not in ranked_object_ids)
    ranked_items.extend(ordered[MAX_CANDIDATES:])

    top_source = (
        ranked_items[0].source_info if isinstance(ranked_items[0].source_info, dict) else {}
    )
    logger.info(
        "semantic scene candidates reranked: "
        f"query={query!r}, candidates={len(selected)}, negatives={len(negatives)}, "
        f"references={len(references)}, "
        f"top_score={top_source.get('semantic_score', 'n/a')}"
    )
    return ranked_items
