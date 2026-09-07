from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# semantic_ranker.py: allow ChatGPT RETRY_SEARCH to supply explicit anti-concepts.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/semantic_ranker.py",
    """    reference_items: List[MaterialInfo] | None = None,\n) -> List[MaterialInfo]:\n""",
    """    reference_items: List[MaterialInfo] | None = None,\n    extra_negative_queries: list[str] | None = None,\n) -> List[MaterialInfo]:\n""",
)
replace_once(
    "app/services/semantic_ranker.py",
    """    negatives = build_negative_queries(query)\n    positive_query = build_positive_query(query) or str(query or \"\").strip()\n""",
    """    negatives: list[str] = []\n    # Manual GPT review anti-concepts take precedence over deterministic rules: the\n    # reviewer has inspected the actual failed candidates and can name exactly what\n    # should be avoided (for example \"startup screen\" or \"empty office\").\n    for value in list(extra_negative_queries or []) + build_negative_queries(query):\n        normalized = \" \".join(str(value or \"\").split())\n        if not normalized or normalized in negatives:\n            continue\n        negatives.append(normalized)\n        if len(negatives) >= MAX_NEGATIVE_QUERIES:\n            break\n    positive_query = build_positive_query(query) or str(query or \"\").strip()\n""",
)


# ---------------------------------------------------------------------------
# strict_scene.py: persist the exact ranked candidate pool used for each scene.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/strict_scene.py",
    "from app.services import semantic_ranker, subtitle, task_artifacts\n",
    "from app.services import gpt_visual_review, semantic_ranker, subtitle, task_artifacts\n",
)
replace_once(
    "app/services/strict_scene.py",
    """        bridge_ranked: list[MaterialInfo] = []\n        bridge_selected = False\n""",
    """        bridge_ranked: list[MaterialInfo] = []\n        bridge_selected = False\n        review_ranked: list[MaterialInfo] = []\n""",
)
replace_once(
    "app/services/strict_scene.py",
    """                bridge_ranked = semantic_ranker.rank_materials(\n                    semantic_query,\n                    bridge_raw,\n                    enabled=True,\n                    reference_items=list(selected_materials),\n                )\n                scene[\"semantic_bridge\"] = True\n""",
    """                bridge_ranked = semantic_ranker.rank_materials(\n                    semantic_query,\n                    bridge_raw,\n                    enabled=True,\n                    reference_items=list(selected_materials),\n                )\n                review_ranked = list(bridge_ranked)\n                scene[\"semantic_bridge\"] = True\n""",
)
replace_once(
    "app/services/strict_scene.py",
    """            for query_index in query_indexes:\n                query = terms[query_index]\n                for item in candidates_for_scene(scene, query):\n                    result = try_candidate(item, allow_reuse=False)\n""",
    """            for query_index in query_indexes:\n                query = terms[query_index]\n                ranked_candidates = candidates_for_scene(scene, query)\n                review_ranked = list(ranked_candidates)\n                for item in ranked_candidates:\n                    result = try_candidate(item, allow_reuse=False)\n""",
)
replace_once(
    "app/services/strict_scene.py",
    """            for query_index in query_indexes:\n                query = terms[query_index]\n                for item in candidates_for_scene(scene, query):\n                    result = try_candidate(item, allow_reuse=True)\n""",
    """            for query_index in query_indexes:\n                query = terms[query_index]\n                ranked_candidates = candidates_for_scene(scene, query)\n                review_ranked = list(ranked_candidates)\n                for item in ranked_candidates:\n                    result = try_candidate(item, allow_reuse=True)\n""",
)
replace_once(
    "app/services/strict_scene.py",
    """        if reused:\n            logger.warning(\n""",
    """        gpt_visual_review.record_scene_candidates(\n            task_id=task_id,\n            scene=scene,\n            ranked_items=review_ranked or [selected_item],\n            selected_item=selected_item,\n            selected_path=saved_path,\n            selected_query=selected_query,\n        )\n\n        if reused:\n            logger.warning(\n""",
)


# ---------------------------------------------------------------------------
# Main.py: expose the review page directly from each completed task row.
# ---------------------------------------------------------------------------
replace_once(
    "webui/Main.py",
    """            has_restore_data = os.path.isfile(\n                os.path.join(task[\"task_path\"], \"script.json\")\n            )\n            safe_task_key = \"\".join(ch if ch.isalnum() else \"_\" for ch in task_id)[:40]\n""",
    """            has_restore_data = os.path.isfile(\n                os.path.join(task[\"task_path\"], \"script.json\")\n            )\n            has_gpt_review_data = os.path.isfile(\n                os.path.join(task[\"task_path\"], \"gpt_review_registry.json\")\n            )\n            safe_task_key = \"\".join(ch if ch.isalnum() else \"_\" for ch in task_id)[:40]\n""",
)
replace_once(
    "webui/Main.py",
    """                action_cols = row_cols[4].columns(\n                    4,\n                    vertical_alignment=\"center\",\n                    gap=\"small\",\n                )\n""",
    """                action_cols = row_cols[4].columns(\n                    5,\n                    vertical_alignment=\"center\",\n                    gap=\"small\",\n                )\n""",
)
replace_once(
    "webui/Main.py",
    """                with action_cols[3]:\n                    delete_label = tr(\"Delete Task\")\n""",
    """                with action_cols[3]:\n                    review_label = \"GPT Visual Review\"\n                    if st.button(\n                        review_label,\n                        key=f\"gpt_review_task_{key_prefix}_{task_id}\",\n                        use_container_width=True,\n                        icon=\":material/rate_review:\",\n                        help=review_label,\n                        disabled=is_processing or not has_gpt_review_data,\n                    ):\n                        st.session_state[\"gpt_visual_review_task_id\"] = task_id\n                        st.switch_page(\"pages/GPT_Visual_Review.py\")\n\n                with action_cols[4]:\n                    delete_label = tr(\"Delete Task\")\n""",
)

print("manual GPT visual-review integration patch applied")
