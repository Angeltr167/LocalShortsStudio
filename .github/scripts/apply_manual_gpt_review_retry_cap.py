from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor in {path}, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "app/services/gpt_visual_review.py",
    "MAX_AVOID_CHARS = 120\n_REVIEW_MARGIN_THRESHOLD = 0.045\n",
    "MAX_AVOID_CHARS = 120\nMAX_RETRY_ROUNDS = 3\n_REVIEW_MARGIN_THRESHOLD = 0.045\n",
)

replace_once(
    "app/services/gpt_visual_review.py",
    '        "current_candidate_id": str(scene.get("current_candidate_id") or ""),\n        "review_required": bool(scene.get("review_required")),\n',
    '        "current_candidate_id": str(scene.get("current_candidate_id") or ""),\n        "retry_round": int(scene.get("retry_round") or 0),\n        "max_retry_rounds": MAX_RETRY_ROUNDS,\n        "retry_exhausted": int(scene.get("retry_round") or 0) >= MAX_RETRY_ROUNDS,\n        "review_required": bool(scene.get("review_required")),\n',
)

replace_once(
    "app/services/gpt_visual_review.py",
    'Allowed actions: keep, select, retry_search. Never invent candidate IDs. For\\nretry_search, describe visible stock-footage content and list misleading visuals in avoid.\\nReturn one valid JSON object with schema_version=1 and the exact task_id.\\n"""\n',
    'Allowed actions: keep, select, retry_search. Never invent candidate IDs. For\\nretry_search, describe visible stock-footage content and list misleading visuals in avoid.\\n\\nRETRY LIMIT: each scene can request at most 3 retry searches. The manifest includes\\nretry_round, max_retry_rounds, and retry_exhausted. If retry_exhausted=true,\\nRETRY_SEARCH IS FORBIDDEN: choose KEEP when current_candidate_id is available, otherwise\\nSELECT the best supplied candidate. Do not keep asking for a perfect literal stock clip\\nwhen the provider cannot supply one; prefer the clearest understandable fallback.\\n\\nReturn one valid JSON object with schema_version=1 and the exact task_id.\\n"""\n',
)

replace_once(
    "app/services/gpt_visual_review.py",
    '        scene = scene_map[scene_number]\n        candidate_ids = {\n',
    '        scene = scene_map[scene_number]\n        if action == "retry_search" and int(scene.get("retry_round") or 0) >= MAX_RETRY_ROUNDS:\n            raise VisualReviewError(\n                f"scene {scene_number} exhausted its {MAX_RETRY_ROUNDS} retry searches; "\n                "choose KEEP or SELECT from the supplied candidates"\n            )\n        candidate_ids = {\n',
)

replace_once(
    "app/services/gpt_visual_review.py",
    '    retry_round = int(scene.get("retry_round") or 0) + 1\n    candidates = [\n        _candidate_payload(\n            f"S{int(scene.get(\'scene\') or 0):02d}-R{retry_round}-C{index}",\n            item,\n        )\n        for index, item in enumerate(ranked[:MAX_REVIEW_CANDIDATES], start=1)\n    ]\n    if not candidates:\n        raise VisualReviewError(f"retry ranking produced no candidates for scene {scene.get(\'scene\')}")\n    scene["candidates"] = candidates\n    # After RETRY_SEARCH there is intentionally no current candidate among the new\n    # options.  The next review round must SELECT one; KEEP is therefore invalid until\n    # a new current candidate is established.\n    scene["current_candidate_id"] = ""\n    scene["retry_round"] = retry_round\n',
    '    retry_round = int(scene.get("retry_round") or 0) + 1\n    if retry_round > MAX_RETRY_ROUNDS:\n        raise VisualReviewError(\n            f"scene {scene.get(\'scene\')} exhausted its {MAX_RETRY_ROUNDS} retry searches"\n        )\n\n    existing_candidates = list(scene.get("candidates") or [])\n    current_candidate_id = str(scene.get("current_candidate_id") or "")\n    current_candidate = next(\n        (\n            candidate\n            for candidate in existing_candidates\n            if str(candidate.get("candidate_id") or "") == current_candidate_id\n        ),\n        None,\n    )\n    new_candidates = [\n        _candidate_payload(\n            f"S{int(scene.get(\'scene\') or 0):02d}-R{retry_round}-C{index}",\n            item,\n        )\n        for index, item in enumerate(ranked[:MAX_REVIEW_CANDIDATES], start=1)\n    ]\n    if not new_candidates:\n        raise VisualReviewError(f"retry ranking produced no candidates for scene {scene.get(\'scene\')}")\n\n    # Never throw away the visual currently used by the rendered video.  Keeping it in\n    # the next candidate set lets ChatGPT fall back to KEEP when a retry search is worse.\n    candidates: list[dict[str, Any]] = []\n    seen_assets: set[tuple[str, str, str]] = set()\n    if current_candidate is not None:\n        candidates.append(current_candidate)\n        seen_assets.add(\n            (\n                str(current_candidate.get("provider") or ""),\n                str(current_candidate.get("asset_id") or ""),\n                str(current_candidate.get("download_url") or ""),\n            )\n        )\n    for candidate in new_candidates:\n        identity = (\n            str(candidate.get("provider") or ""),\n            str(candidate.get("asset_id") or ""),\n            str(candidate.get("download_url") or ""),\n        )\n        if identity in seen_assets:\n            continue\n        candidates.append(candidate)\n        seen_assets.add(identity)\n        if len(candidates) >= MAX_REVIEW_CANDIDATES:\n            break\n\n    scene["candidates"] = candidates\n    scene["retry_round"] = retry_round\n',
)

replace_once(
    "app/services/gpt_visual_review.py",
    '    scene["review_required"] = True\n    scene["review_reasons"] = ["gpt_requested_retry_search"]\n',
    '    scene["review_required"] = True\n    scene["retry_exhausted"] = retry_round >= MAX_RETRY_ROUNDS\n    scene["review_reasons"] = [\n        "gpt_retry_limit_reached"\n        if scene["retry_exhausted"]\n        else "gpt_requested_retry_search"\n    ]\n',
)

replace_once(
    "app/services/gpt_visual_review.py",
    '        "retry_pending_count": sum(bool(scene.get("retry_pending")) for scene in scenes),\n        "package_revision": int(registry.get("package_revision") or 0),\n',
    '        "retry_pending_count": sum(bool(scene.get("retry_pending")) for scene in scenes),\n        "retry_exhausted_count": sum(\n            bool(scene.get("retry_pending"))\n            and int(scene.get("retry_round") or 0) >= MAX_RETRY_ROUNDS\n            for scene in scenes\n        ),\n        "max_retry_rounds": MAX_RETRY_ROUNDS,\n        "package_revision": int(registry.get("package_revision") or 0),\n',
)

replace_once(
    "webui/pages/GPT_Visual_Review.py",
    'scope_label = st.radio(\n    "Package scope",\n    ["Recommended scenes", "All scenes"],\n    horizontal=True,\n',
    'scope_options = ["Recommended scenes", "All scenes"]\nif summary["retry_pending_count"]:\n    scope_options.insert(0, "Retry scenes only")\nscope_label = st.radio(\n    "Package scope",\n    scope_options,\n    horizontal=True,\n',
)

replace_once(
    "webui/pages/GPT_Visual_Review.py",
    'scope = "recommended" if scope_label == "Recommended scenes" else "all"\n',
    'scope = (\n    "retry"\n    if scope_label == "Retry scenes only"\n    else "recommended"\n    if scope_label == "Recommended scenes"\n    else "all"\n)\n\nif summary.get("retry_exhausted_count"):\n    st.warning(\n        f"{summary[\'retry_exhausted_count\']} retry scene(s) reached the hard limit of "\n        f"{summary[\'max_retry_rounds\']} searches. The next GPT review must KEEP or SELECT; "\n        "another RETRY_SEARCH will be rejected."\n    )\n',
)

print("manual GPT visual review retry cap applied")
