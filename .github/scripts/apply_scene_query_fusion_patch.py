from pathlib import Path

STRICT = Path("app/services/strict_scene.py")
RANKER = Path("app/services/semantic_ranker.py")
TESTS = Path("test/services/test_semantic_scene_ranking.py")

strict = STRICT.read_text(encoding="utf-8")
ranker = RANKER.read_text(encoding="utf-8")
tests = TESTS.read_text(encoding="utf-8")

if "_SCENE_QUERY_RRF_K = 60.0" in strict:
    print("scene query fusion patch already applied")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# semantic_ranker.py: keep negation as contrast, not as the positive CLIP text.
# ---------------------------------------------------------------------------
ranker = ranker.replace("import math\n", "import math\nimport re\n", 1)

negative_marker = '''def reference_preview_urls(items: List[MaterialInfo]) -> list[str]:\n'''
positive_helper = r'''def build_positive_query(query: str) -> str:
    """Return a CLIP-positive description with explicit anti-concepts removed.

    CLIP-style retrieval is unreliable when a desired visual is phrased through
    negation (for example ``no phone``).  The original query is still used to
    build ``negative_queries``; this function only rewrites the positive side so
    the service sees what should actually be visible.
    """
    text = " ".join(str(query or "").split())
    if not text:
        return ""

    substitutions = (
        (r"\bdo\s+not\s+disturb\b", "phone face down silent notifications"),
        (r"\bwithout\s+(?:a\s+|the\s+)?phone\b", ""),
        (r"\bno\s+phone\b", ""),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip(" ,;.-")


'''
assert negative_marker in ranker, "semantic ranker insertion marker missing"
ranker = ranker.replace(negative_marker, positive_helper + negative_marker, 1)

old_payload = '''    negatives = build_negative_queries(query)\n    references = reference_preview_urls(reference_items or [])\n    base_url = _base_url()\n    payload = {\n        "query": str(query or "").strip(),\n        "negative_queries": negatives,\n'''
new_payload = '''    negatives = build_negative_queries(query)\n    positive_query = build_positive_query(query) or str(query or "").strip()\n    references = reference_preview_urls(reference_items or [])\n    base_url = _base_url()\n    payload = {\n        "query": positive_query,\n        "negative_queries": negatives,\n'''
assert old_payload in ranker, "semantic ranker payload marker missing"
ranker = ranker.replace(old_payload, new_payload, 1)

# ---------------------------------------------------------------------------
# strict_scene.py: deterministic scene-specific query rewriting + RRF fusion.
# ---------------------------------------------------------------------------
helper_marker = '''def _fallback_query_indexes(primary_index: int, query_count: int) -> list[int]:\n'''
helpers = r'''_SCENE_QUERY_RRF_K = 60.0
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


'''
assert helper_marker in strict, "strict scene helper insertion marker missing"
strict = strict.replace(helper_marker, helpers + helper_marker, 1)

old_helpers = '''    def merged_raw_candidates(queries: list[str]) -> List[MaterialInfo]:\n        """Merge provider searches while deduplicating alternate renditions/assets."""\n\n        merged: list[MaterialInfo] = []\n        seen: set[str] = set()\n        for query in queries:\n            for item in raw_candidates_for(query):\n                identity = _material_identity(item)\n                if identity in seen:\n                    continue\n                seen.add(identity)\n                merged.append(item)\n        return merged\n\n    def candidates_for(query: str) -> List[MaterialInfo]:\n        return semantic_ranker.rank_materials(\n            query,\n            raw_candidates_for(query),\n            enabled=semantic_scene_ranking,\n            reference_items=list(selected_materials),\n        )\n'''
new_helpers = '''    def fused_raw_candidates(\n        specs: list[tuple[str, float, str]],\n    ) -> List[MaterialInfo]:\n        ranked_results = [\n            (query, weight, raw_candidates_for(query))\n            for query, weight, _ in specs\n        ]\n        return _fuse_provider_rankings(ranked_results)\n\n    def record_scene_queries(\n        scene: dict[str, Any],\n        specs: list[tuple[str, float, str]],\n    ) -> None:\n        existing = scene.setdefault("search_queries", [])\n        known = {str(value.get("query") or "").lower() for value in existing if isinstance(value, dict)}\n        for query, weight, kind in specs:\n            if query.lower() in known:\n                continue\n            existing.append(\n                {"query": query, "weight": round(float(weight), 3), "kind": kind}\n            )\n            known.add(query.lower())\n\n    def candidates_for_scene(\n        scene: dict[str, Any],\n        anchor_query: str,\n    ) -> List[MaterialInfo]:\n        if not semantic_scene_ranking:\n            return raw_candidates_for(anchor_query)\n\n        narration = str(scene.get("narration_text") or "")\n        specs = _query_specs_for_scene(narration, anchor_query)\n        record_scene_queries(scene, specs)\n        semantic_query = _build_semantic_scene_query(narration, anchor_query)\n        scene["semantic_query"] = semantic_query\n        return semantic_ranker.rank_materials(\n            semantic_query,\n            fused_raw_candidates(specs),\n            enabled=True,\n            reference_items=list(selected_materials),\n        )\n'''
assert old_helpers in strict, "strict scene local helper marker missing"
strict = strict.replace(old_helpers, new_helpers, 1)

old_bridge = '''                bridge_raw = merged_raw_candidates(\n                    [\n                        bridge_query,\n                        terms[primary_index],\n                        terms[bridge_to_index],\n                    ]\n                )\n                bridge_ranked = semantic_ranker.rank_materials(\n                    bridge_query,\n                    bridge_raw,\n                    enabled=True,\n                    reference_items=list(selected_materials),\n                )\n                scene["semantic_bridge"] = True\n                scene["semantic_query"] = bridge_query\n'''
new_bridge = '''                narration = str(scene.get("narration_text") or "")\n                bridge_specs = _query_specs_for_scene(narration, bridge_query)\n                existing_bridge_queries = {query.lower() for query, _, _ in bridge_specs}\n                for query, weight, kind in (\n                    (terms[primary_index], _BRIDGE_NEIGHBOR_WEIGHT, "bridge_current"),\n                    (terms[bridge_to_index], _BRIDGE_NEIGHBOR_WEIGHT, "bridge_next"),\n                ):\n                    safe_query = _provider_safe_text(query)\n                    if safe_query and safe_query.lower() not in existing_bridge_queries:\n                        bridge_specs.append((safe_query, weight, kind))\n                        existing_bridge_queries.add(safe_query.lower())\n                record_scene_queries(scene, bridge_specs)\n                bridge_raw = fused_raw_candidates(bridge_specs)\n                semantic_query = _build_semantic_scene_query(narration, bridge_query)\n                bridge_ranked = semantic_ranker.rank_materials(\n                    semantic_query,\n                    bridge_raw,\n                    enabled=True,\n                    reference_items=list(selected_materials),\n                )\n                scene["semantic_bridge"] = True\n                scene["semantic_query"] = semantic_query\n'''
assert old_bridge in strict, "bridge ranking marker missing"
strict = strict.replace(old_bridge, new_bridge, 1)

old_first_pass = '''        # First pass: only unused source assets.\n        if selected is None:\n            for query_index in query_indexes:\n                query = terms[query_index]\n                for item in candidates_for(query):\n                    result = try_candidate(item, allow_reuse=False)\n                    if result is None:\n                        continue\n                    selected = result\n                    selected_item = item\n                    selected_query = query\n                    break\n                if selected is not None:\n                    break\n'''
new_first_pass = '''        # First pass: only unused source assets.  Semantic mode searches a\n        # scene-specific rewrite plus the original anchor and fuses both provider\n        # rankings before OpenCLIP.  Strict-only mode remains byte-for-byte\n        # equivalent in intent: a single anchor query at a time.\n        if selected is None:\n            for query_index in query_indexes:\n                query = terms[query_index]\n                for item in candidates_for_scene(scene, query):\n                    result = try_candidate(item, allow_reuse=False)\n                    if result is None:\n                        continue\n                    selected = result\n                    selected_item = item\n                    source = item.source_info if isinstance(item.source_info, dict) else {}\n                    selected_query = str(source.get("search_term") or query)\n                    break\n                if selected is not None:\n                    break\n'''
assert old_first_pass in strict, "first pass marker missing"
strict = strict.replace(old_first_pass, new_first_pass, 1)

old_last_resort = '''        # Last resort: allow a source to repeat rather than leaving the timeline short.\n        if selected is None:\n            for query_index in query_indexes:\n                query = terms[query_index]\n                for item in candidates_for(query):\n                    result = try_candidate(item, allow_reuse=True)\n                    if result is None:\n                        continue\n                    selected = result\n                    selected_item = item\n                    selected_query = query\n                    break\n                if selected is not None:\n                    break\n'''
new_last_resort = '''        # Last resort: allow a source to repeat rather than leaving the timeline short.\n        if selected is None:\n            for query_index in query_indexes:\n                query = terms[query_index]\n                for item in candidates_for_scene(scene, query):\n                    result = try_candidate(item, allow_reuse=True)\n                    if result is None:\n                        continue\n                    selected = result\n                    selected_item = item\n                    source = item.source_info if isinstance(item.source_info, dict) else {}\n                    selected_query = str(source.get("search_term") or query)\n                    break\n                if selected is not None:\n                    break\n'''
assert old_last_resort in strict, "last resort marker missing"
strict = strict.replace(old_last_resort, new_last_resort, 1)

# ---------------------------------------------------------------------------
# Tests: provider-safe rewrite, RRF fusion, and positive/negative separation.
# ---------------------------------------------------------------------------
append_tests = r'''


def test_scene_specific_query_combines_timed_narration_and_visual_anchor():
    query = strict_scene._build_scene_specific_query(
        "Then sit down and finish the report at your laptop before checking messages.",
        "focused person deep work laptop no phone",
    )
    assert query.startswith("focused person deep work laptop")
    assert "report" in query
    assert "phone" not in query
    assert " no " not in f" {query} "
    assert len(query.split()) <= strict_scene._PROVIDER_QUERY_MAX_WORDS


def test_provider_rrf_rewards_candidate_returned_by_multiple_scene_phrasings():
    repeated = _item("repeated", "https://images.pexels.com/repeated.jpg")
    specific_only = _item("specific-only", "https://images.pexels.com/specific.jpg")
    anchor_only = _item("anchor-only", "https://images.pexels.com/anchor.jpg")

    fused = strict_scene._fuse_provider_rankings(
        [
            ("specific", 1.0, [specific_only, repeated]),
            ("anchor", 1.0, [anchor_only, repeated]),
        ]
    )

    assert fused[0].source_info["asset_id"] == "repeated"
    assert fused[0].source_info["provider_query_hits"] == 2
    assert fused[0].source_info["provider_fusion_score"] > fused[1].source_info["provider_fusion_score"]


def test_rank_materials_moves_no_phone_constraint_to_negative_side():
    first = _item("first-positive", "https://images.pexels.com/first-positive.jpg")
    second = _item("second-positive", "https://images.pexels.com/second-positive.jpg")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "model": "ViT-B-32/test",
        "ranked": [
            {"id": "pexels:first-positive", "score": 0.5},
            {"id": "pexels:second-positive", "score": 0.4},
        ],
    }
    session = Mock()
    session.post.return_value = response

    with patch("app.services.semantic_ranker._local_session", return_value=session):
        semantic_ranker.rank_materials(
            "calm focused person deep work laptop no phone",
            [first, second],
            enabled=True,
        )

    payload = session.post.call_args.kwargs["json"]
    assert "no phone" not in payload["query"].lower()
    assert "focused person" in payload["query"].lower()
    assert "person holding a smartphone" in payload["negative_queries"]


def test_scene_query_specs_keep_specific_rewrite_and_original_anchor():
    specs = strict_scene._query_specs_for_scene(
        "a half-written email is still waiting to be finished",
        "half written email on laptop screen office",
    )
    assert specs[0][2] == "scene_specific"
    assert specs[-1][2] == "anchor"
    assert specs[0][0] != specs[-1][0]
    assert "finished" in specs[0][0]
'''
if "test_scene_specific_query_combines_timed_narration_and_visual_anchor" not in tests:
    tests += append_tests

STRICT.write_text(strict, encoding="utf-8")
RANKER.write_text(ranker, encoding="utf-8")
TESTS.write_text(tests, encoding="utf-8")
print("scene query fusion patch applied")
