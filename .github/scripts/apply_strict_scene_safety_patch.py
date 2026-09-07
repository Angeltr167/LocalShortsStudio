from pathlib import Path


path = Path("app/services/task.py")
text = path.read_text(encoding="utf-8")

marker = "\n\ndef _get_video_music_prompt(params: VideoParams) -> str:\n"
if text.count(marker) != 1:
    raise RuntimeError("unexpected task.py music-prompt marker")
helper = '''

_STOCK_VIDEO_SOURCES = frozenset({"pexels", "pixabay", "coverr"})


def _strict_scene_matching_enabled(params: VideoParams) -> bool:
    """Strict scene assignment only applies to searchable stock-video providers."""
    return bool(
        params.strict_scene_matching
        and params.video_source in _STOCK_VIDEO_SOURCES
    )


def _get_video_music_prompt(params: VideoParams) -> str:
'''
text = text.replace(marker, helper, 1)

old_download = '''                video_concat_mode=(
                    VideoConcatMode.sequential
                    if (
                        params.match_materials_to_script
                        or params.strict_scene_matching
                    )
                    else params.video_concat_mode
                ),
                audio_duration=(
                    audio_duration
                    if params.strict_scene_matching
                    else audio_duration * params.video_count
                ),
                max_clip_duration=params.video_clip_duration,
                match_script_order=params.match_materials_to_script,
                strict_scene_matching=params.strict_scene_matching,
'''
new_download = '''                video_concat_mode=(
                    VideoConcatMode.sequential
                    if (
                        params.match_materials_to_script
                        or _strict_scene_matching_enabled(params)
                    )
                    else params.video_concat_mode
                ),
                audio_duration=(
                    audio_duration
                    if _strict_scene_matching_enabled(params)
                    else audio_duration * params.video_count
                ),
                max_clip_duration=params.video_clip_duration,
                match_script_order=params.match_materials_to_script,
                strict_scene_matching=_strict_scene_matching_enabled(params),
'''
if text.count(old_download) != 1:
    raise RuntimeError("unexpected patched stock download block")
text = text.replace(old_download, new_download, 1)

old_final = (
    "    if params.match_materials_to_script or params.strict_scene_matching:\n"
    "        video_concat_mode = VideoConcatMode.sequential\n"
)
new_final = (
    "    if params.match_materials_to_script or _strict_scene_matching_enabled(params):\n"
    "        video_concat_mode = VideoConcatMode.sequential\n"
)
if text.count(old_final) != 1:
    raise RuntimeError("unexpected patched final concat block")
text = text.replace(old_final, new_final, 1)

path.write_text(text, encoding="utf-8")
print("strict scene stock-source safety patch applied")
