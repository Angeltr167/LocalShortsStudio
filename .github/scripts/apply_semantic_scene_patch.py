from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one match in {path}, found {count}: {old[:180]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Data model: semantic ranking is opt-in and only meaningful with Strict Scene.
# ---------------------------------------------------------------------------
replace_once(
    "app/models/schema.py",
    "    strict_scene_matching: bool = False\n"
    "    video_count: int = Field(default=1, ge=1)\n",
    "    strict_scene_matching: bool = False\n"
    "    # Optional local OpenCLIP reranking of stock preview images. This only\n"
    "    # changes candidate ordering inside Strict Scene Matching.\n"
    "    semantic_scene_ranking: bool = False\n"
    "    video_count: int = Field(default=1, ge=1)\n",
)


# ---------------------------------------------------------------------------
# Provider search results: retain only transient preview URLs needed for local
# ranking. They are intentionally not included by _material_source_record.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/material.py",
    "                        \"rendition\": {\n"
    "                            \"id\": (\n"
    "                                str(video.get(\"id\"))\n"
    "                                if video.get(\"id\") is not None\n"
    "                                else None\n"
    "                            ),\n"
    "                            \"width\": w,\n"
    "                            \"height\": h,\n"
    "                        },\n"
    "                    }\n"
    "                    video_items.append(item)\n"
    "                    break\n"
    "        return video_items\n"
    "    except Exception as e:\n"
    "        logger.error(\n"
    "            \"pexels video search failed: \"\n",
    "                        \"rendition\": {\n"
    "                            \"id\": (\n"
    "                                str(video.get(\"id\"))\n"
    "                                if video.get(\"id\") is not None\n"
    "                                else None\n"
    "                            ),\n"
    "                            \"width\": w,\n"
    "                            \"height\": h,\n"
    "                        },\n"
    "                        \"preview_images\": [\n"
    "                            str(picture.get(\"picture\"))\n"
    "                            for picture in (v.get(\"video_pictures\") or [])\n"
    "                            if isinstance(picture, dict) and picture.get(\"picture\")\n"
    "                        ],\n"
    "                    }\n"
    "                    video_items.append(item)\n"
    "                    break\n"
    "        return video_items\n"
    "    except Exception as e:\n"
    "        logger.error(\n"
    "            \"pexels video search failed: \"\n",
)

replace_once(
    "app/services/material.py",
    "                        \"rendition\": {\n"
    "                            \"id\": video_type,\n"
    "                            \"width\": w,\n"
    "                            \"height\": video.get(\"height\"),\n"
    "                        },\n"
    "                    }\n"
    "                    video_items.append(item)\n"
    "                    break\n"
    "        return video_items\n"
    "    except Exception as e:\n"
    "        error_message = _redact_request_error(e, api_key)\n",
    "                        \"rendition\": {\n"
    "                            \"id\": video_type,\n"
    "                            \"width\": w,\n"
    "                            \"height\": video.get(\"height\"),\n"
    "                        },\n"
    "                        \"preview_images\": (\n"
    "                            [str(video.get(\"thumbnail\"))]\n"
    "                            if video.get(\"thumbnail\")\n"
    "                            else []\n"
    "                        ),\n"
    "                    }\n"
    "                    video_items.append(item)\n"
    "                    break\n"
    "        return video_items\n"
    "    except Exception as e:\n"
    "        error_message = _redact_request_error(e, api_key)\n",
)

replace_once(
    "app/services/material.py",
    "                \"rendition\": {\n"
    "                    \"id\": \"mp4_download\",\n"
    "                    \"width\": v.get(\"max_width\"),\n"
    "                    \"height\": v.get(\"max_height\"),\n"
    "                },\n"
    "            }\n"
    "            video_items.append(item)\n"
    "        return video_items\n"
    "    except Exception as e:\n"
    "        logger.error(\n"
    "            \"coverr video search failed: \"\n",
    "                \"rendition\": {\n"
    "                    \"id\": \"mp4_download\",\n"
    "                    \"width\": v.get(\"max_width\"),\n"
    "                    \"height\": v.get(\"max_height\"),\n"
    "                },\n"
    "                \"preview_images\": [\n"
    "                    str(url)\n"
    "                    for url in (v.get(\"thumbnail\"), v.get(\"poster\"))\n"
    "                    if url\n"
    "                ],\n"
    "            }\n"
    "            video_items.append(item)\n"
    "        return video_items\n"
    "    except Exception as e:\n"
    "        logger.error(\n"
    "            \"coverr video search failed: \"\n",
)

replace_once(
    "app/services/material.py",
    "    match_script_order: bool = False,\n"
    "    strict_scene_matching: bool = False,\n"
    ") -> List[str]:\n",
    "    match_script_order: bool = False,\n"
    "    strict_scene_matching: bool = False,\n"
    "    semantic_scene_ranking: bool = False,\n"
    ") -> List[str]:\n",
)
replace_once(
    "app/services/material.py",
    "            material_directory=material_directory,\n"
    "        )\n\n"
    "    if match_script_order:\n",
    "            material_directory=material_directory,\n"
    "            semantic_scene_ranking=semantic_scene_ranking,\n"
    "        )\n\n"
    "    if match_script_order:\n",
)


# ---------------------------------------------------------------------------
# Strict scene selector: rerank each provider candidate list before attempting
# downloads. Failure in the optional ranker preserves provider ordering.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/strict_scene.py",
    "from app.services import task_artifacts\n",
    "from app.services import semantic_ranker, task_artifacts\n",
)
replace_once(
    "app/services/strict_scene.py",
    "    material_directory: str,\n"
    ") -> List[str]:\n",
    "    material_directory: str,\n"
    "    semantic_scene_ranking: bool = False,\n"
    ") -> List[str]:\n",
)
replace_once(
    "app/services/strict_scene.py",
    "            candidate_cache[query] = list(items)\n"
    "            logger.info(\n"
    "                f\"found {len(candidate_cache[query])} strict candidates for {query!r}\"\n"
    "            )\n",
    "            ranked_items = semantic_ranker.rank_materials(\n"
    "                query,\n"
    "                list(items),\n"
    "                enabled=semantic_scene_ranking,\n"
    "            )\n"
    "            candidate_cache[query] = ranked_items\n"
    "            logger.info(\n"
    "                f\"found {len(candidate_cache[query])} strict candidates for {query!r}\"\n"
    "            )\n",
)
replace_once(
    "app/services/strict_scene.py",
    "                \"reused\": bool(reused),\n"
    "            }\n"
    "        )\n",
    "                \"reused\": bool(reused),\n"
    "                \"semantic_score\": source.get(\"semantic_score\"),\n"
    "                \"semantic_rank\": source.get(\"semantic_rank\"),\n"
    "                \"semantic_ranker_model\": source.get(\"semantic_ranker_model\"),\n"
    "            }\n"
    "        )\n",
)


# ---------------------------------------------------------------------------
# Task orchestration: semantic ranking is valid only when strict stock matching
# is active, including API callers that bypass the WebUI.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/task.py",
    "def _get_video_music_prompt(params: VideoParams) -> str:\n",
    "def _semantic_scene_ranking_enabled(params: VideoParams) -> bool:\n"
    "    \"\"\"Semantic reranking is a sub-mode of strict stock scene matching.\"\"\"\n"
    "    return bool(\n"
    "        params.semantic_scene_ranking\n"
    "        and _strict_scene_matching_enabled(params)\n"
    "    )\n\n\n"
    "def _get_video_music_prompt(params: VideoParams) -> str:\n",
)
replace_once(
    "app/services/task.py",
    "                strict_scene_matching=_strict_scene_matching_enabled(params),\n"
    "            )\n",
    "                strict_scene_matching=_strict_scene_matching_enabled(params),\n"
    "                semantic_scene_ranking=_semantic_scene_ranking_enabled(params),\n"
    "            )\n",
)


# ---------------------------------------------------------------------------
# WebUI: persist/restore the toggle and keep it subordinate to Strict Scene.
# ---------------------------------------------------------------------------
replace_once(
    "webui/Main.py",
    "        \"strict_scene_matching\": bool(\n"
    "            config.app.get(\"strict_scene_matching\", False)\n"
    "        ),\n",
    "        \"strict_scene_matching\": bool(\n"
    "            config.app.get(\"strict_scene_matching\", False)\n"
    "        ),\n"
    "        \"semantic_scene_ranking\": bool(\n"
    "            config.app.get(\"semantic_scene_ranking\", False)\n"
    "        ),\n",
)
replace_once(
    "webui/Main.py",
    "    st.session_state[\"strict_scene_matching\"] = restored_strict_scene_matching\n"
    "    st.session_state[\"match_materials_to_script\"] = bool(\n",
    "    st.session_state[\"strict_scene_matching\"] = restored_strict_scene_matching\n"
    "    st.session_state[\"semantic_scene_ranking\"] = bool(\n"
    "        params.get(\"semantic_scene_ranking\", False)\n"
    "        and restored_strict_scene_matching\n"
    "    )\n"
    "    st.session_state[\"match_materials_to_script\"] = bool(\n",
)
replace_once(
    "webui/Main.py",
    "    else:\n"
    "        previous_match = st.session_state.pop(previous_key, None)\n"
    "        if previous_match is not None:\n"
    "            st.session_state[\"match_materials_to_script\"] = bool(previous_match)\n"
    "    sync_script_order_concat_mode()\n",
    "    else:\n"
    "        st.session_state[\"semantic_scene_ranking\"] = False\n"
    "        previous_match = st.session_state.pop(previous_key, None)\n"
    "        if previous_match is not None:\n"
    "            st.session_state[\"match_materials_to_script\"] = bool(previous_match)\n"
    "    sync_script_order_concat_mode()\n",
)
replace_once(
    "webui/Main.py",
    "            if params.strict_scene_matching:\n"
    "                params.match_materials_to_script = True\n\n"
    "            _set_runtime_config(\n"
    "                \"app\",\n"
    "                \"match_materials_to_script\",\n"
    "                params.match_materials_to_script,\n"
    "            )\n",
    "            if params.strict_scene_matching:\n"
    "                params.match_materials_to_script = True\n"
    "            elif st.session_state.get(\"semantic_scene_ranking\", False):\n"
    "                st.session_state[\"semantic_scene_ranking\"] = False\n\n"
    "            params.semantic_scene_ranking = st.checkbox(\n"
    "                tr(\"Semantic Scene Ranking\"),\n"
    "                help=tr(\"Semantic Scene Ranking Help\"),\n"
    "                key=\"semantic_scene_ranking\",\n"
    "                disabled=(\n"
    "                    not strict_scene_supported\n"
    "                    or not params.strict_scene_matching\n"
    "                ),\n"
    "            )\n"
    "            params.semantic_scene_ranking = bool(\n"
    "                params.semantic_scene_ranking and params.strict_scene_matching\n"
    "            )\n\n"
    "            _set_runtime_config(\n"
    "                \"app\",\n"
    "                \"match_materials_to_script\",\n"
    "                params.match_materials_to_script,\n"
    "            )\n",
)
replace_once(
    "webui/Main.py",
    "            _set_runtime_config(\n"
    "                \"app\",\n"
    "                \"strict_scene_matching\",\n"
    "                params.strict_scene_matching,\n"
    "            )\n"
    "            # Ordered modes derive sequential composition and must not overwrite the\n",
    "            _set_runtime_config(\n"
    "                \"app\",\n"
    "                \"strict_scene_matching\",\n"
    "                params.strict_scene_matching,\n"
    "            )\n"
    "            _set_runtime_config(\n"
    "                \"app\",\n"
    "                \"semantic_scene_ranking\",\n"
    "                params.semantic_scene_ranking,\n"
    "            )\n"
    "            # Ordered modes derive sequential composition and must not overwrite the\n",
)
replace_once(
    "webui/Main.py",
    "    params.strict_scene_matching = bool(\n"
    "        st.session_state.get(\"strict_scene_matching\", False)\n"
    "    )\n"
    "    params.match_materials_to_script = bool(\n",
    "    params.strict_scene_matching = bool(\n"
    "        st.session_state.get(\"strict_scene_matching\", False)\n"
    "    )\n"
    "    params.semantic_scene_ranking = bool(\n"
    "        st.session_state.get(\"semantic_scene_ranking\", False)\n"
    "        and params.strict_scene_matching\n"
    "    )\n"
    "    params.match_materials_to_script = bool(\n",
)


# ---------------------------------------------------------------------------
# Config defaults.
# ---------------------------------------------------------------------------
replace_once(
    "config.example.toml",
    "strict_scene_matching = false\n\n"
    "# API key lists support key rotation. Use straight ASCII double quotes and\n",
    "strict_scene_matching = false\n\n"
    "# Optional local OpenCLIP reranking for Strict Scene Matching. The service\n"
    "# runs separately so its PyTorch environment cannot disturb the main app.\n"
    "semantic_scene_ranking = false\n"
    "semantic_ranker_base_url = \"http://127.0.0.1:4124\"\n\n"
    "# API key lists support key rotation. Use straight ASCII double quotes and\n",
)

print("semantic scene ranking patch applied successfully")
