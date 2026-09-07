from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one match in {path}, found {count}: {old[:160]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(path: str, start_marker: str, end_marker: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    start_count = text.count(start_marker)
    if start_count != 1:
        raise RuntimeError(
            f"expected exactly one start marker in {path}, found {start_count}: "
            f"{start_marker[:160]!r}"
        )
    start = text.index(start_marker)
    try:
        end = text.index(end_marker, start + len(start_marker))
    except ValueError as exc:
        raise RuntimeError(
            f"end marker not found in {path}: {end_marker[:160]!r}"
        ) from exc
    target.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


# ---------------------------------------------------------------------------
# Data model: opt-in and backwards compatible by default.
# ---------------------------------------------------------------------------
replace_once(
    "app/models/schema.py",
    "    match_materials_to_script: bool = False\n"
    "    video_count: int = Field(default=1, ge=1)\n",
    "    match_materials_to_script: bool = False\n"
    "    # Opt-in stock-footage assignment that maps ordered visual queries to\n"
    "    # clip-sized timeline scenes. Disabled by default for compatibility.\n"
    "    strict_scene_matching: bool = False\n"
    "    video_count: int = Field(default=1, ge=1)\n",
)


# ---------------------------------------------------------------------------
# Material service: route only stock sources through strict scene assignment.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/material.py",
    "from app.services import (\n    material_cache,\n",
    "from app.services import (\n    material_cache,\n    strict_scene,\n",
)
replace_once(
    "app/services/material.py",
    "    max_clip_duration: int = 5,\n"
    "    match_script_order: bool = False,\n"
    ") -> List[str]:\n",
    "    max_clip_duration: int = 5,\n"
    "    match_script_order: bool = False,\n"
    "    strict_scene_matching: bool = False,\n"
    ") -> List[str]:\n",
)
replace_once(
    "app/services/material.py",
    "    if match_script_order:\n"
    "        return _download_videos_by_script_order(\n",
    "    if strict_scene_matching and source in {\"pexels\", \"pixabay\", \"coverr\"}:\n"
    "        return strict_scene.download_videos_by_scene_queries(\n"
    "            task_id=task_id,\n"
    "            search_terms=search_terms,\n"
    "            search_videos=search_videos,\n"
    "            save_video=save_video,\n"
    "            source_record=_material_source_record,\n"
    "            persist_sources=_persist_material_sources,\n"
    "            redact_error=_redact_request_error,\n"
    "            video_aspect=video_aspect,\n"
    "            audio_duration=audio_duration,\n"
    "            max_clip_duration=max_clip_duration,\n"
    "            material_directory=material_directory,\n"
    "        )\n\n"
    "    if match_script_order:\n"
    "        return _download_videos_by_script_order(\n",
)


# ---------------------------------------------------------------------------
# Task orchestration: keep strict stock clips sequential and use one narration
# duration, not video_count multiplied duration, for the scene plan.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/task.py",
    "                video_concat_mode=(\n"
    "                    VideoConcatMode.sequential\n"
    "                    if params.match_materials_to_script\n"
    "                    else params.video_concat_mode\n"
    "                ),\n"
    "                audio_duration=audio_duration * params.video_count,\n"
    "                max_clip_duration=params.video_clip_duration,\n"
    "                match_script_order=params.match_materials_to_script,\n",
    "                video_concat_mode=(\n"
    "                    VideoConcatMode.sequential\n"
    "                    if (\n"
    "                        params.match_materials_to_script\n"
    "                        or params.strict_scene_matching\n"
    "                    )\n"
    "                    else params.video_concat_mode\n"
    "                ),\n"
    "                audio_duration=(\n"
    "                    audio_duration\n"
    "                    if params.strict_scene_matching\n"
    "                    else audio_duration * params.video_count\n"
    "                ),\n"
    "                max_clip_duration=params.video_clip_duration,\n"
    "                match_script_order=params.match_materials_to_script,\n"
    "                strict_scene_matching=params.strict_scene_matching,\n",
)
replace_once(
    "app/services/task.py",
    "    if params.match_materials_to_script:\n"
    "        video_concat_mode = VideoConcatMode.sequential\n",
    "    if params.match_materials_to_script or params.strict_scene_matching:\n"
    "        video_concat_mode = VideoConcatMode.sequential\n",
)


# ---------------------------------------------------------------------------
# WebUI session state / restore paths. Presets automatically include the new
# VideoParams field, so restoring it here is enough to make presets portable.
# ---------------------------------------------------------------------------
replace_once(
    "webui/Main.py",
    "        \"match_materials_to_script\": bool(\n"
    "            config.app.get(\"match_materials_to_script\", False)\n"
    "        ),\n",
    "        \"match_materials_to_script\": bool(\n"
    "            config.app.get(\"match_materials_to_script\", False)\n"
    "        ),\n"
    "        \"strict_scene_matching\": bool(\n"
    "            config.app.get(\"strict_scene_matching\", False)\n"
    "        ),\n",
)
replace_once(
    "webui/Main.py",
    "    st.session_state[\"match_materials_to_script\"] = bool(\n"
    "        params.get(\"match_materials_to_script\", False)\n"
    "    )\n",
    "    restored_strict_scene_matching = bool(\n"
    "        params.get(\"strict_scene_matching\", False)\n"
    "    )\n"
    "    st.session_state[\"strict_scene_matching\"] = restored_strict_scene_matching\n"
    "    st.session_state[\"match_materials_to_script\"] = bool(\n"
    "        params.get(\"match_materials_to_script\", False)\n"
    "        or restored_strict_scene_matching\n"
    "    )\n",
)

sync_replacement = '''def sync_script_order_concat_mode():
    """Force sequential composition while either ordered mode is enabled."""
    widget_key = localized_widget_key("video_concat_mode_select")
    previous_key = "video_concat_mode_before_script_order_match"
    ordered_mode_enabled = bool(
        st.session_state.get("match_materials_to_script", False)
        or st.session_state.get("strict_scene_matching", False)
    )

    if ordered_mode_enabled:
        current_mode = st.session_state.get(widget_key, VideoConcatMode.random.value)
        if current_mode != VideoConcatMode.sequential.value:
            st.session_state[previous_key] = current_mode
        st.session_state[widget_key] = VideoConcatMode.sequential.value
        return

    previous_mode = st.session_state.pop(previous_key, None)
    if previous_mode in {
        VideoConcatMode.sequential.value,
        VideoConcatMode.random.value,
    }:
        st.session_state[widget_key] = previous_mode


def sync_strict_scene_matching():
    """Strict mode implies script-order matching and restores the previous toggle."""
    previous_key = "match_materials_before_strict_scene_matching"
    strict_enabled = bool(st.session_state.get("strict_scene_matching", False))
    if strict_enabled:
        st.session_state.setdefault(
            previous_key,
            bool(st.session_state.get("match_materials_to_script", False)),
        )
        st.session_state["match_materials_to_script"] = True
    else:
        previous_match = st.session_state.pop(previous_key, None)
        if previous_match is not None:
            st.session_state["match_materials_to_script"] = bool(previous_match)
    sync_script_order_concat_mode()


'''
replace_between(
    "webui/Main.py",
    "def sync_script_order_concat_mode():\n",
    "def reset_script_system_prompt():\n",
    sync_replacement,
)

# Concat select remains disabled when either ordered mode is active.
replace_once(
    "webui/Main.py",
    "                disabled=bool(st.session_state.get(\"match_materials_to_script\", False)),\n",
    "                disabled=bool(\n"
    "                    st.session_state.get(\"match_materials_to_script\", False)\n"
    "                    or st.session_state.get(\"strict_scene_matching\", False)\n"
    "                ),\n",
)

video_toggle_replacement = '''            strict_scene_supported = (
                params.video_source in VIDEO_SOURCE_GROUPS["stock_video"]
            )
            if (
                not strict_scene_supported
                and st.session_state.get("strict_scene_matching", False)
            ):
                st.session_state["strict_scene_matching"] = False
                sync_strict_scene_matching()

            params.match_materials_to_script = st.checkbox(
                tr("Match Materials to Script Order"),
                help=tr("Match Materials to Script Order Help"),
                key="match_materials_to_script",
                on_change=sync_script_order_concat_mode,
                disabled=bool(st.session_state.get("strict_scene_matching", False)),
            )
            params.strict_scene_matching = st.checkbox(
                tr("Strict Scene Matching"),
                help=tr("Strict Scene Matching Help"),
                key="strict_scene_matching",
                on_change=sync_strict_scene_matching,
                disabled=not strict_scene_supported,
            )
            if params.strict_scene_matching:
                params.match_materials_to_script = True

            _set_runtime_config(
                "app",
                "match_materials_to_script",
                params.match_materials_to_script,
            )
            _set_runtime_config(
                "app",
                "strict_scene_matching",
                params.strict_scene_matching,
            )
            # Ordered modes derive sequential composition and must not overwrite the
            # normal random/sequential preference while they are active.
            if not (
                params.match_materials_to_script or params.strict_scene_matching
            ):
                _set_runtime_config(
                    "ui", "video_concat_mode", params.video_concat_mode.value
                )

'''
replace_between(
    "webui/Main.py",
    "            params.match_materials_to_script = st.checkbox(\n",
    "            # 视频转场模式\n",
    video_toggle_replacement,
)

replace_once(
    "webui/Main.py",
    "    params = VideoParams(video_subject=\"\")\n"
    "    params.match_materials_to_script = bool(\n"
    "        st.session_state.get(\"match_materials_to_script\", False)\n"
    "    )\n",
    "    params = VideoParams(video_subject=\"\")\n"
    "    params.strict_scene_matching = bool(\n"
    "        st.session_state.get(\"strict_scene_matching\", False)\n"
    "    )\n"
    "    params.match_materials_to_script = bool(\n"
    "        st.session_state.get(\"match_materials_to_script\", False)\n"
    "        or params.strict_scene_matching\n"
    "    )\n",
)


# ---------------------------------------------------------------------------
# Config and translations.
# ---------------------------------------------------------------------------
replace_once(
    "config.example.toml",
    "video_source = \"pexels\"\n\n# API key lists support key rotation.\n",
    "video_source = \"pexels\"\n\n"
    "# Opt-in scene assignment for stock footage. Ordered keywords become scene\n"
    "# queries; one clip is chosen for each clip-sized narration slot.\n"
    "strict_scene_matching = false\n\n"
    "# API key lists support key rotation.\n",
)
replace_once(
    "webui/i18n/en.json",
    '    "Match Materials to Script Order Help": "Generate keywords and match materials in script order so visuals follow the current voiceover.",\n',
    '    "Match Materials to Script Order Help": "Generate keywords and match materials in script order so visuals follow the current voiceover.",\n'
    '    "Strict Scene Matching": "Strict Scene Matching",\n'
    '    "Strict Scene Matching Help": "Treat ordered stock-video keywords as scene queries, assign one clip per timeline scene, avoid duplicate sources when possible, and preserve narrative order. Available for Pexels, Pixabay, and Coverr.",\n',
)
replace_once(
    "webui/i18n/es.json",
    '    "Match Materials to Script Order Help": "Genera términos y ordena el material según el guion para que las imágenes sigan la narración.",\n',
    '    "Match Materials to Script Order Help": "Genera términos y ordena el material según el guion para que las imágenes sigan la narración.",\n'
    '    "Strict Scene Matching": "Coincidencia estricta por escenas",\n'
    '    "Strict Scene Matching Help": "Trata las palabras clave ordenadas como búsquedas de escena, asigna un clip por escena de la línea de tiempo, evita fuentes duplicadas cuando sea posible y conserva el orden narrativo. Disponible para Pexels, Pixabay y Coverr.",\n',
)

print("strict scene patch applied successfully")
