from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one anchor in {path}, found {count}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema: AI cartoon source is a first-class task mode with bounded controls.
# ---------------------------------------------------------------------------
replace_once(
    "app/models/schema.py",
    """    semantic_scene_ranking: bool = False\n    video_count: int = Field(default=1, ge=1)\n""",
    """    semantic_scene_ranking: bool = False\n    # Local procedural 2D animation mode. The LLM only chooses from a bounded\n    # scene vocabulary; drawing/rendering remains deterministic and local.\n    cartoon_ai_director: bool = True\n    cartoon_lip_sync: Literal[\"auto\", \"heuristic\", \"rhubarb\"] = \"auto\"\n    cartoon_fps: int = Field(default=24, ge=12, le=30)\n    video_count: int = Field(default=1, ge=1)\n""",
)


# ---------------------------------------------------------------------------
# LLM: tiny public adapter for scene-direction JSON, reusing existing provider config.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/llm.py",
    """def test_connection() -> tuple[bool, str, float]:\n""",
    """def generate_cartoon_direction(prompt: str, app_config=None) -> str:\n    \"\"\"Generate a bounded cartoon scene plan with the configured text LLM.\n\n    The caller validates every enum and provides a deterministic fallback, so this\n    helper intentionally stays thin and never grants the model tool or file access.\n    A second attempt helps local Ollama models that occasionally return an empty\n    reasoning-only response on their first generation.\n    \"\"\"\n    last_response = \"\"\n    for attempt in range(2):\n        last_response = (\n            _generate_response(prompt=prompt)\n            if app_config is None\n            else _generate_response(prompt=prompt, app_config=app_config)\n        )\n        if last_response and not last_response.startswith(\"Error:\"):\n            return last_response\n        if attempt == 0:\n            logger.warning(\"AI cartoon director returned no usable plan; retrying once\")\n    return last_response\n\n\ndef test_connection() -> tuple[bool, str, float]:\n""",
)


# ---------------------------------------------------------------------------
# Task pipeline: skip stock terms and render a full-length cartoon material locally.
# ---------------------------------------------------------------------------
replace_once(
    "app/services/task.py",
    """import re\nimport socket\n""",
    """import re\nimport shutil\nimport socket\n""",
)
replace_once(
    "app/services/task.py",
    """    elevenlabs_music,\n    llm,\n    loomloom,\n""",
    """    cartoon_engine,\n    elevenlabs_music,\n    llm,\n    loomloom,\n""",
)
replace_once(
    "app/services/task.py",
    """def get_video_materials(\n    task_id,\n    params,\n    video_terms,\n    audio_duration,\n    subtitle_path: str = \"\",\n    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,\n):\n    if params.video_source == \"local\":\n""",
    """def get_video_materials(\n    task_id,\n    params,\n    video_terms,\n    audio_duration,\n    subtitle_path: str = \"\",\n    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,\n    video_script: str = \"\",\n    audio_file: str = \"\",\n):\n    if params.video_source == \"ai_cartoon\":\n        logger.info(\"\\n\\n## rendering local AI-directed cartoon material\")\n        try:\n            cartoon_file = cartoon_engine.render_cartoon_material(\n                task_id=task_id,\n                params=params,\n                video_script=video_script,\n                audio_file=audio_file,\n                audio_duration=audio_duration,\n                subtitle_path=subtitle_path,\n            )\n        except cartoon_engine.CartoonRenderError as exc:\n            _mark_task_failed(task_id, \"materials\", str(exc))\n            return None\n        return [cartoon_file] if cartoon_file else None\n    if params.video_source == \"local\":\n""",
)
replace_once(
    "app/services/task.py",
    """    if params.video_source != \"local\":\n        video_terms = generate_terms(task_id, params, video_script)\n""",
    """    if params.video_source not in {\"local\", \"ai_cartoon\"}:\n        video_terms = generate_terms(task_id, params, video_script)\n""",
)
replace_once(
    "app/services/task.py",
    """        subtitle_path=subtitle_path,\n        loomloom_video_request=loomloom_video_request,\n    )\n""",
    """        subtitle_path=subtitle_path,\n        loomloom_video_request=loomloom_video_request,\n        video_script=video_script,\n        audio_file=audio_file,\n    )\n""",
)
replace_once(
    "app/services/task.py",
    """    if params.match_materials_to_script or _strict_scene_matching_enabled(params):\n        video_concat_mode = VideoConcatMode.sequential\n    elif params.video_count == 1:\n""",
    """    if params.video_source == \"ai_cartoon\":\n        # The material already contains the narration-aware animation timeline.\n        # Randomization, transitions or clip-speed changes would destroy sync.\n        video_concat_mode = VideoConcatMode.sequential\n    elif params.match_materials_to_script or _strict_scene_matching_enabled(params):\n        video_concat_mode = VideoConcatMode.sequential\n    elif params.video_count == 1:\n""",
)
replace_once(
    "app/services/task.py",
    """    video_transition_mode = params.video_transition_mode\n\n    _progress = 50\n""",
    """    video_transition_mode = (\n        None if params.video_source == \"ai_cartoon\" else params.video_transition_mode\n    )\n    clip_speed = 1.0 if params.video_source == \"ai_cartoon\" else params.video_clip_speed\n\n    _progress = 50\n""",
)
replace_once(
    "app/services/task.py",
    """        logger.info(f\"\\n\\n## combining video: {index} => {combined_video_path}\")\n        video.combine_videos(\n            combined_video_path=combined_video_path,\n            video_paths=downloaded_videos,\n            audio_file=audio_file,\n            video_aspect=params.video_aspect,\n            video_fit_mode=params.video_fit_mode,\n            video_concat_mode=video_concat_mode,\n            video_transition_mode=video_transition_mode,\n            max_clip_duration=params.video_clip_duration,\n            threads=params.n_threads,\n            clip_speed=params.video_clip_speed,\n        )\n""",
    """        logger.info(f\"\\n\\n## combining video: {index} => {combined_video_path}\")\n        if params.video_source == \"ai_cartoon\":\n            # The procedural renderer already emits one exact full-length timeline.\n            # Sending it through combine_videos() would split at video_clip_duration\n            # and, in sequential mode, keep only the first segment before looping it.\n            if len(downloaded_videos) != 1 or not path.isfile(downloaded_videos[0]):\n                raise cartoon_engine.CartoonRenderError(\n                    \"AI cartoon mode requires exactly one rendered timeline material\"\n                )\n            shutil.copy2(downloaded_videos[0], combined_video_path)\n        else:\n            video.combine_videos(\n                combined_video_path=combined_video_path,\n                video_paths=downloaded_videos,\n                audio_file=audio_file,\n                video_aspect=params.video_aspect,\n                video_fit_mode=params.video_fit_mode,\n                video_concat_mode=video_concat_mode,\n                video_transition_mode=video_transition_mode,\n                max_clip_duration=params.video_clip_duration,\n                threads=params.n_threads,\n                clip_speed=clip_speed,\n            )\n""",
)


# ---------------------------------------------------------------------------
# Web UI: expose the new source and animation-specific controls.
# ---------------------------------------------------------------------------
replace_once(
    "webui/Main.py",
    """VIDEO_SOURCE_GROUPS = {\n    \"stock_video\": (\"pexels\", \"pixabay\", \"coverr\"),\n""",
    """VIDEO_SOURCE_GROUPS = {\n    \"stock_video\": (\"pexels\", \"pixabay\", \"coverr\"),\n    \"animation\": (\"ai_cartoon\",),\n""",
)
replace_once(
    "webui/Main.py",
    """                \"openai_image\": tr(\"OpenAI Compatible Text-to-Image\"),\n                \"local\": tr(\"Local file\"),\n""",
    """                \"openai_image\": tr(\"OpenAI Compatible Text-to-Image\"),\n                \"ai_cartoon\": \"AI Cartoon · Doodle Podcast\",\n                \"local\": tr(\"Local file\"),\n""",
)
replace_once(
    "webui/Main.py",
    """                    (tr(\"Stock Video\"), VIDEO_SOURCE_GROUPS[\"stock_video\"]),\n                    (tr(\"AI Video\"), VIDEO_SOURCE_GROUPS[\"ai_video\"]),\n""",
    """                    (tr(\"Stock Video\"), VIDEO_SOURCE_GROUPS[\"stock_video\"]),\n                    (\"AI Animation\", VIDEO_SOURCE_GROUPS[\"animation\"]),\n                    (tr(\"AI Video\"), VIDEO_SOURCE_GROUPS[\"ai_video\"]),\n""",
)
replace_once(
    "webui/Main.py",
    """            if params.video_source == \"metaso_minimax\":\n                st.caption(tr(\"Metaso MiniMax H3 Help\"))\n            if params.video_source == \"local\":\n""",
    """            if params.video_source == \"metaso_minimax\":\n                st.caption(tr(\"Metaso MiniMax H3 Help\"))\n            if params.video_source == \"ai_cartoon\":\n                st.caption(\n                    \"Local 2D animation: the configured LLM directs scenes while an \"\n                    \"original procedural doodle renderer keeps characters consistent. \"\n                    \"No stock-footage or text-to-video API is used. Video keywords are ignored.\"\n                )\n            if params.video_source == \"local\":\n""",
)
replace_once(
    "webui/Main.py",
    """            strict_scene_supported = (\n                params.video_source in VIDEO_SOURCE_GROUPS[\"stock_video\"]\n            )\n""",
    """            strict_scene_supported = (\n                params.video_source in VIDEO_SOURCE_GROUPS[\"stock_video\"]\n            )\n            if params.video_source == \"ai_cartoon\":\n                # Stock-order controls do not participate in the procedural animation\n                # timeline. Clear stale session values when switching sources.\n                st.session_state[\"match_materials_to_script\"] = False\n                st.session_state[\"strict_scene_matching\"] = False\n                st.session_state[\"semantic_scene_ranking\"] = False\n""",
)
replace_once(
    "webui/Main.py",
    """                disabled=bool(st.session_state.get(\"strict_scene_matching\", False)),\n            )\n""",
    """                disabled=bool(\n                    st.session_state.get(\"strict_scene_matching\", False)\n                    or params.video_source == \"ai_cartoon\"\n                ),\n            )\n""",
)
replace_once(
    "webui/Main.py",
    """            params.video_clip_speed = st.slider(\n                tr(\"Clip Speed\"),\n                min_value=0.5,\n                max_value=2.0,\n                step=0.05,\n                format=\"%.2fx\",\n                key=clip_speed_key,\n                help=tr(\"Clip Speed Help\"),\n            )\n""",
    """            if params.video_source == \"ai_cartoon\":\n                st.session_state[clip_speed_key] = 1.0\n            params.video_clip_speed = st.slider(\n                tr(\"Clip Speed\"),\n                min_value=0.5,\n                max_value=2.0,\n                step=0.05,\n                format=\"%.2fx\",\n                key=clip_speed_key,\n                help=(\n                    \"Cartoon timing is locked to the narration.\"\n                    if params.video_source == \"ai_cartoon\"\n                    else tr(\"Clip Speed Help\")\n                ),\n                disabled=params.video_source == \"ai_cartoon\",\n            )\n""",
)
replace_once(
    "webui/Main.py",
    """            video_count_options = [1, 2, 3, 4, 5]\n            params.video_count = stable_selectbox(\n""",
    """            video_count_options = (\n                [1] if params.video_source == \"ai_cartoon\" else [1, 2, 3, 4, 5]\n            )\n            params.video_count = stable_selectbox(\n""",
)
replace_once(
    "webui/Main.py",
    """            if params.video_source == \"metaso_minimax\":\n                _render_metaso_minimax_video_settings(params)\n    return uploaded_files\n\n\ndef _render_wavespeed_video_settings(params):\n""",
    """            if params.video_source == \"metaso_minimax\":\n                _render_metaso_minimax_video_settings(params)\n            if params.video_source == \"ai_cartoon\":\n                _render_ai_cartoon_settings(params)\n    return uploaded_files\n\n\ndef _render_ai_cartoon_settings(params):\n    \"\"\"Controls for the local procedural doodle animation source.\"\"\"\n    st.markdown(\"**AI Cartoon Director**\")\n    params.cartoon_ai_director = st.checkbox(\n        \"Use configured LLM as scene director\",\n        value=_saved_ui_bool(\"cartoon_ai_director\", True),\n        key=\"cartoon_ai_director\",\n        help=(\n            \"The LLM chooses only validated layouts, expressions and explanatory \"\n            \"overlays. If it fails, LocalShortsStudio automatically uses a deterministic plan.\"\n        ),\n    )\n    _set_runtime_config(\"ui\", \"cartoon_ai_director\", params.cartoon_ai_director)\n\n    lip_options = [\"auto\", \"heuristic\", \"rhubarb\"]\n    params.cartoon_lip_sync = stable_selectbox(\n        \"Lip sync\",\n        options=lip_options,\n        default_value=_saved_ui_choice(\"cartoon_lip_sync\", lip_options, \"auto\"),\n        key=\"cartoon_lip_sync_select\",\n        format_func=lambda value: {\n            \"auto\": \"Auto (Rhubarb when installed, local fallback otherwise)\",\n            \"heuristic\": \"Built-in local mouth animation\",\n            \"rhubarb\": \"Require Rhubarb phoneme lip sync\",\n        }[value],\n    )\n    _set_runtime_config(\"ui\", \"cartoon_lip_sync\", params.cartoon_lip_sync)\n\n    fps_options = [18, 24, 30]\n    params.cartoon_fps = stable_selectbox(\n        \"Cartoon FPS\",\n        options=fps_options,\n        default_value=_saved_ui_choice(\"cartoon_fps\", fps_options, 24),\n        key=\"cartoon_fps_select\",\n        help=\"24 FPS is recommended. 18 FPS is faster; 30 FPS is smoother but heavier.\",\n    )\n    _set_runtime_config(\"ui\", \"cartoon_fps\", params.cartoon_fps)\n\n    rhubarb_path = st.text_input(\n        \"Rhubarb executable (optional)\",\n        value=str(config.app.get(\"rhubarb_path\", \"\") or \"\"),\n        placeholder=r\"C:\\Tools\\rhubarb\\rhubarb.exe\",\n        help=(\n            \"Leave blank for the built-in lip-sync fallback. In Auto mode a configured \"\n            \"Rhubarb binary is detected automatically.\"\n        ),\n        key=\"rhubarb_path_input\",\n    )\n    _set_runtime_config(\"app\", \"rhubarb_path\", rhubarb_path.strip())\n    st.caption(\n        \"The first version uses an original midnight-podcast doodle set with two stable \"\n        \"characters, microphones, reactions and animated concept cards.\"\n    )\n\n\ndef _render_wavespeed_video_settings(params):\n""",
)

print("AI cartoon engine integration patch applied")
