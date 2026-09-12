from pathlib import Path


path = Path("webui/Main.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one UI polish anchor, found {count}: {old[:120]!r}")
    text = text.replace(old, new, 1)


replace_once(
    '''            sync_script_order_concat_mode()
            selected_concat_mode = stable_selectbox(
''',
    '''            sync_script_order_concat_mode()
            if params.video_source == "ai_cartoon":
                st.session_state["video_concat_mode_select"] = VideoConcatMode.sequential.value
            selected_concat_mode = stable_selectbox(
''',
)
replace_once(
    '''                disabled=bool(
                    st.session_state.get("match_materials_to_script", False)
                    or st.session_state.get("strict_scene_matching", False)
                ),
''',
    '''                disabled=bool(
                    st.session_state.get("match_materials_to_script", False)
                    or st.session_state.get("strict_scene_matching", False)
                    or params.video_source == "ai_cartoon"
                ),
''',
)
replace_once(
    '''            selected_transition_mode = stable_selectbox(
                tr("Video Transition Mode"),
''',
    '''            if params.video_source == "ai_cartoon":
                st.session_state["video_transition_mode_select"] = VideoTransitionMode.none.value
            selected_transition_mode = stable_selectbox(
                tr("Video Transition Mode"),
''',
)
replace_once(
    '''                format_func=lambda value: dict(
                    (v, label) for label, v in video_transition_modes
                )[value],
            )
            params.video_transition_mode = VideoTransitionMode(selected_transition_mode)
''',
    '''                format_func=lambda value: dict(
                    (v, label) for label, v in video_transition_modes
                )[value],
                disabled=params.video_source == "ai_cartoon",
                help=(
                    "Scene changes are authored inside the cartoon timeline."
                    if params.video_source == "ai_cartoon"
                    else None
                ),
            )
            params.video_transition_mode = VideoTransitionMode(selected_transition_mode)
''',
)
replace_once(
    '''            params.video_clip_duration = stable_selectbox(
                tr("Clip Duration"),
                options=video_clip_durations,
''',
    '''            params.video_clip_duration = stable_selectbox(
                tr("Clip Duration"),
                options=video_clip_durations,
''',
)
replace_once(
    '''                key="video_clip_duration_select",
                help=tr("Clip Duration Help"),
            )
''',
    '''                key="video_clip_duration_select",
                help=(
                    "Not used by AI Cartoon; narration timing controls scene duration."
                    if params.video_source == "ai_cartoon"
                    else tr("Clip Duration Help")
                ),
                disabled=params.video_source == "ai_cartoon",
            )
''',
)

path.write_text(text, encoding="utf-8")
print("AI cartoon UI polish patch applied")
