from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.services import gpt_visual_review  # noqa: E402

st.set_page_config(
    page_title="GPT Visual Review",
    page_icon="🎬",
    layout="wide",
)

style_file = Path(root_dir) / "webui" / "styles.css"
if style_file.is_file():
    st.markdown(
        f"<style>{style_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


st.title("🎬 GPT Visual Review")
st.caption(
    "Manual ChatGPT Plus review: export candidate contact sheets, paste the JSON "
    "decisions back here, and rebuild only the visual timeline. No OpenAI API call is made."
)

if st.button("← Back to generator", use_container_width=False):
    st.switch_page("Main.py")


tasks = gpt_visual_review.list_review_tasks()
if not tasks:
    st.info(
        "No review-ready tasks yet. Generate a stock-footage video with Strict Scene "
        "Matching enabled first."
    )
    st.stop()

requested_task_id = str(st.session_state.get("gpt_visual_review_task_id") or "")
options = [task["task_id"] for task in tasks]
default_index = options.index(requested_task_id) if requested_task_id in options else 0
selected_task_id = st.selectbox(
    "Task",
    options,
    index=default_index,
    format_func=lambda task_id: next(
        (
            f"{task['subject']} · {task_id[:8]}"
            for task in tasks
            if task["task_id"] == task_id
        ),
        task_id,
    ),
)
st.session_state["gpt_visual_review_task_id"] = selected_task_id

try:
    summary = gpt_visual_review.review_summary(selected_task_id)
except gpt_visual_review.VisualReviewError as exc:
    st.error(str(exc))
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Scenes", summary["scene_count"])
metric_cols[1].metric("Recommended review", summary["review_required_count"])
metric_cols[2].metric("Retry pending", summary["retry_pending_count"])
metric_cols[3].metric("Review builds", summary["review_revision"])

if summary["current_video"] and Path(summary["current_video"]).is_file():
    with st.expander("Current rendered video", expanded=False):
        st.video(summary["current_video"])

st.divider()
st.subheader("1. Export package for your dedicated ChatGPT review chat")
scope_label = st.radio(
    "Package scope",
    ["Recommended scenes", "All scenes"],
    horizontal=True,
    help=(
        "Recommended scenes uses score margin, low-confidence, reuse and bridge signals. "
        "All scenes is useful for a full audit."
    ),
)
scope = "recommended" if scope_label == "Recommended scenes" else "all"

if st.button(
    "Build GPT Review Package",
    type="primary",
    use_container_width=True,
):
    try:
        package = gpt_visual_review.export_review_package(selected_task_id, scope=scope)
        st.session_state["gpt_visual_review_last_zip"] = package["zip_path"]
        st.success(
            f"Package v{package['revision']} created for scenes: "
            + ", ".join(str(value) for value in package["included_scenes"])
        )
    except gpt_visual_review.VisualReviewError as exc:
        st.error(str(exc))

latest_zip = str(
    st.session_state.get("gpt_visual_review_last_zip")
    or summary.get("last_package_zip")
    or ""
)
if latest_zip and Path(latest_zip).is_file():
    zip_path = Path(latest_zip)
    st.download_button(
        "Download GPT Review Package",
        data=zip_path.read_bytes(),
        file_name=zip_path.name,
        mime="application/zip",
        use_container_width=True,
    )
    st.caption(
        "Upload this ZIP to the dedicated visual-director chat you already configured. "
        "The package contains the current video, review_manifest.json and labeled scene sheets."
    )

st.divider()
st.subheader("2. Import ChatGPT decisions")
json_upload = st.file_uploader(
    "Upload the JSON returned by ChatGPT",
    type=["json"],
    key=f"gpt_review_json_{selected_task_id}",
)
json_text = st.text_area(
    "Or paste the JSON object",
    height=260,
    placeholder='{"schema_version": 1, "task_id": "...", "decisions": [...] }',
    key=f"gpt_review_text_{selected_task_id}",
)


def _decision_payload():
    if json_upload is not None:
        try:
            text = json_upload.getvalue().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise gpt_visual_review.VisualReviewError("uploaded JSON must be UTF-8") from exc
    else:
        text = json_text
    if not str(text or "").strip():
        raise gpt_visual_review.VisualReviewError("paste or upload the GPT decision JSON first")
    return gpt_visual_review.parse_decision_json(text)


validate_col, apply_col = st.columns(2)
if validate_col.button("Validate Decisions", use_container_width=True):
    try:
        payload = _decision_payload()
        normalized = gpt_visual_review.validate_decisions(selected_task_id, payload)
        st.success(f"Valid review: {len(normalized)} scene decisions are ready to apply.")
        with st.expander("Validated decisions"):
            st.json(normalized)
    except gpt_visual_review.VisualReviewError as exc:
        st.error(str(exc))

if apply_col.button(
    "Apply Decisions",
    type="primary",
    use_container_width=True,
    help=(
        "SELECT downloads only the chosen replacements. RETRY_SEARCH creates a second-pass "
        "package. When no retries remain, the existing master audio/subtitles are reused "
        "to rebuild the video."
    ),
):
    try:
        payload = _decision_payload()
        with st.spinner("Applying visual-review decisions..."):
            result = gpt_visual_review.apply_decisions(selected_task_id, payload)
        if result["status"] == "retry_required":
            retry_zip = Path(result["package_zip"])
            st.warning(
                "ChatGPT requested new searches for scenes: "
                + ", ".join(str(value) for value in result["retry_scenes"])
                + ". A second-pass package is ready."
            )
            if retry_zip.is_file():
                st.download_button(
                    "Download Retry Review Package",
                    data=retry_zip.read_bytes(),
                    file_name=retry_zip.name,
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            reviewed_video = Path(result["reviewed_video"])
            st.success(
                f"Reviewed video rebuilt without regenerating script, TTS or subtitles: "
                f"{reviewed_video.name}"
            )
            if reviewed_video.is_file():
                st.video(str(reviewed_video))
                st.download_button(
                    "Download Reviewed Video",
                    data=reviewed_video.read_bytes(),
                    file_name=reviewed_video.name,
                    mime="video/mp4",
                    use_container_width=True,
                )
    except gpt_visual_review.VisualReviewError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)

st.divider()
st.caption(
    "Safety: imported JSON cannot contain commands, arbitrary local paths or URLs. "
    "Candidate IDs must exist in the last package; retry queries are length-limited and "
    "are used only for the task's existing stock provider."
)
