"""Manual ChatGPT visual-review workflow for strict stock-footage tasks.

The service deliberately does not call ChatGPT or any external LLM API.  It records a
bounded candidate set while a task is generated, exports a self-contained review package
for the user to inspect with ChatGPT Plus, validates the returned JSON as untrusted input,
and can rebuild only the visual timeline while reusing the already-rendered master audio
and subtitle timeline.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Any, List
from urllib.parse import urlsplit

import requests
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode, VideoParams
from app.services import task_artifacts
from app.utils import utils

REVIEW_SCHEMA_VERSION = 1
REGISTRY_SCHEMA = "localshortsstudio.gpt-visual-review-registry"
MANIFEST_SCHEMA = "localshortsstudio.gpt-visual-review"
REGISTRY_FILE_NAME = "gpt_review_registry.json"
REVIEW_DIR_NAME = "gpt_review"
MAX_REVIEW_CANDIDATES = 4
MAX_PREVIEWS_PER_CANDIDATE = 8
CONTACT_SHEET_PREVIEWS = 3
MAX_RETRY_QUERY_CHARS = 180
MAX_RETRY_QUERY_WORDS = 24
MAX_AVOID_ITEMS = 8
MAX_AVOID_CHARS = 120
_REVIEW_MARGIN_THRESHOLD = 0.045
_REVIEW_SCORE_THRESHOLD = 0.22
_ALLOWED_ACTIONS = frozenset({"keep", "select", "retry_search"})
_ALLOWED_PREVIEW_HOST_SUFFIXES = (
    "images.pexels.com",
    "cdn.pixabay.com",
    "pixabay.com",
    "coverr.co",
)


class VisualReviewError(RuntimeError):
    """User-facing review workflow error."""


def _safe_task_directory(task_id: str) -> Path:
    normalized = str(task_id or "").strip()
    if not normalized or Path(normalized).name != normalized or normalized in {".", ".."}:
        raise VisualReviewError("invalid task id")
    tasks_root = Path(utils.task_dir()).resolve()
    target = (tasks_root / normalized).resolve()
    try:
        target.relative_to(tasks_root)
    except ValueError as exc:
        raise VisualReviewError("task path is outside the task directory") from exc
    return target


def _registry_path(task_id: str) -> Path:
    return _safe_task_directory(task_id) / REGISTRY_FILE_NAME


def _script_path(task_id: str) -> Path:
    return _safe_task_directory(task_id) / "script.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_registry(task_id: str) -> dict[str, Any]:
    path = _registry_path(task_id)
    if not path.is_file():
        raise VisualReviewError("this task has no GPT visual-review candidate registry")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualReviewError("the GPT visual-review registry is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != REGISTRY_SCHEMA
        or payload.get("schema_version") != REVIEW_SCHEMA_VERSION
        or payload.get("task_id") != str(task_id)
        or not isinstance(payload.get("scenes"), list)
    ):
        raise VisualReviewError("the GPT visual-review registry has an unsupported format")
    return payload


def _load_registry_or_new(task_id: str) -> dict[str, Any]:
    path = _registry_path(task_id)
    if path.is_file():
        return load_registry(task_id)
    return {
        "schema": REGISTRY_SCHEMA,
        "schema_version": REVIEW_SCHEMA_VERSION,
        "task_id": str(task_id),
        "package_revision": 0,
        "review_revision": 0,
        "scenes": [],
    }


def _material_identity(item: MaterialInfo) -> str:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    provider = str(item.provider or source.get("provider") or "unknown")
    asset_id = source.get("asset_id")
    if asset_id not in (None, ""):
        return f"{provider}:asset:{asset_id}"
    return f"{provider}:url:{item.url}"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _preview_urls(item: MaterialInfo) -> list[str]:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    raw = source.get("preview_images")
    values = raw if isinstance(raw, list) else []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
        if len(output) >= MAX_PREVIEWS_PER_CANDIDATE:
            break
    return output


def _candidate_payload(candidate_id: str, item: MaterialInfo) -> dict[str, Any]:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "provider": str(item.provider or source.get("provider") or ""),
        "asset_id": str(source.get("asset_id") or ""),
        "duration": float(item.duration or 0),
        # Private registry fields.  They are intentionally stripped from the exported
        # manifest; SELECT needs the original URL later without exposing signed URLs.
        "download_url": str(item.url or ""),
        "preview_urls": _preview_urls(item),
        "local_path": "",
    }
    for key in (
        "semantic_score",
        "semantic_positive_score",
        "semantic_negative_score",
        "semantic_diversity_similarity",
        "semantic_rank",
        "provider_fusion_score",
        "provider_query_hits",
    ):
        value = source.get(key)
        if value not in (None, ""):
            payload[key] = value
    return payload


def _review_reasons(
    scene: dict[str, Any],
    candidates: list[dict[str, Any]],
    current_candidate_id: str,
) -> list[str]:
    reasons: list[str] = []
    current = next(
        (candidate for candidate in candidates if candidate["candidate_id"] == current_candidate_id),
        None,
    )
    current_score = _finite_number((current or {}).get("semantic_score"))
    scores = [
        score
        for score in (_finite_number(candidate.get("semantic_score")) for candidate in candidates)
        if score is not None
    ]
    if current is None:
        reasons.append("current_candidate_not_in_review_pool")
    if current_score is None:
        reasons.append("semantic_score_unavailable")
    elif current_score < _REVIEW_SCORE_THRESHOLD:
        reasons.append("low_semantic_score")
    if len(scores) >= 2 and abs(scores[0] - scores[1]) < _REVIEW_MARGIN_THRESHOLD:
        reasons.append("narrow_top_score_margin")
    if bool(scene.get("reused")):
        reasons.append("source_reused")
    if bool(scene.get("bridge_selected")):
        reasons.append("semantic_bridge_selected")
    selected_rank = (current or {}).get("semantic_rank")
    try:
        if selected_rank not in (None, "") and int(selected_rank) > 1:
            reasons.append("selected_candidate_not_semantic_rank_one")
    except (TypeError, ValueError):
        pass
    if len(candidates) < 2:
        reasons.append("small_candidate_pool")
    return list(dict.fromkeys(reasons))


def record_scene_candidates(
    *,
    task_id: str,
    scene: dict[str, Any],
    ranked_items: List[MaterialInfo],
    selected_item: MaterialInfo,
    selected_path: str,
    selected_query: str,
) -> bool:
    """Persist a bounded private candidate registry for one selected timeline scene.

    This is deliberately fail-open.  A review artifact is auxiliary metadata and must
    never make normal video generation fail.
    """
    try:
        # Unit tests and third-party direct calls can invoke strict_scene without a task
        # manifest.  Do not create a half-task solely for review metadata.
        if not _script_path(task_id).is_file():
            return False
        registry = _load_registry_or_new(task_id)
        selected_identity = _material_identity(selected_item)
        pool = list(ranked_items[:MAX_REVIEW_CANDIDATES])
        if not any(_material_identity(item) == selected_identity for item in pool):
            if len(pool) >= MAX_REVIEW_CANDIDATES:
                pool[-1] = selected_item
            else:
                pool.append(selected_item)

        scene_number = int(scene.get("scene") or 0)
        candidates: list[dict[str, Any]] = []
        current_candidate_id = ""
        for index, item in enumerate(pool, start=1):
            candidate_id = f"S{scene_number:02d}-C{index}"
            candidate = _candidate_payload(candidate_id, item)
            if _material_identity(item) == selected_identity:
                candidate["local_path"] = str(selected_path or "")
                current_candidate_id = candidate_id
            candidates.append(candidate)

        if not current_candidate_id:
            # The selected item is always appended above, but keep a defensive fallback
            # so a future identity change cannot create an unusable registry.
            candidate_id = f"S{scene_number:02d}-C{len(candidates) + 1}"
            candidate = _candidate_payload(candidate_id, selected_item)
            candidate["local_path"] = str(selected_path or "")
            candidates.append(candidate)
            current_candidate_id = candidate_id

        scene_entry = {
            "scene": scene_number,
            "start": float(scene.get("start") or 0),
            "end": float(scene.get("end") or 0),
            "duration": float(scene.get("duration") or 0),
            "narration": str(scene.get("narration_text") or ""),
            "visual_intent": str(scene.get("query") or ""),
            "semantic_query": str(scene.get("semantic_query") or ""),
            "selected_query": str(selected_query or ""),
            "current_candidate_id": current_candidate_id,
            "current_path": str(selected_path or ""),
            "candidates": candidates,
            "reused": bool(scene.get("reused")),
            "bridge_selected": bool(scene.get("bridge_selected")),
            "retry_pending": False,
        }
        reasons = _review_reasons(scene, candidates, current_candidate_id)
        scene_entry["review_reasons"] = reasons
        scene_entry["review_required"] = bool(reasons)

        scenes = [
            existing
            for existing in registry.get("scenes", [])
            if int(existing.get("scene") or 0) != scene_number
        ]
        scenes.append(scene_entry)
        scenes.sort(key=lambda value: int(value.get("scene") or 0))
        registry["scenes"] = scenes
        _write_json_atomic(_registry_path(task_id), registry)
        task_artifacts.patch_script_data(
            task_id,
            gpt_visual_review={
                "available": True,
                "registry_file": REGISTRY_FILE_NAME,
                "scene_count": len(scenes),
            },
        )
        return True
    except Exception as exc:
        logger.warning(
            "failed to record GPT visual-review candidates: "
            f"task_id={task_id}, scene={scene.get('scene')}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return False


def _manifest_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "candidate_id",
        "provider",
        "asset_id",
        "duration",
        "semantic_score",
        "semantic_positive_score",
        "semantic_negative_score",
        "semantic_diversity_similarity",
        "semantic_rank",
        "provider_fusion_score",
        "provider_query_hits",
    )
    return {key: candidate[key] for key in allowed if key in candidate}


def _current_video(task_dir: Path) -> Path | None:
    reviewed = []
    for candidate in task_dir.glob("final-reviewed-*.mp4"):
        match = re.fullmatch(r"final-reviewed-(\d+)\.mp4", candidate.name)
        if match and candidate.is_file():
            reviewed.append((int(match.group(1)), candidate))
    if reviewed:
        return max(reviewed, key=lambda value: value[0])[1]
    original = task_dir / "final-1.mp4"
    return original if original.is_file() else None


def _allowed_preview_url(value: str) -> bool:
    try:
        parsed = urlsplit(str(value or ""))
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    host = parsed.hostname.lower()
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in _ALLOWED_PREVIEW_HOST_SUFFIXES)


def _download_preview(url: str) -> Image.Image | None:
    if not _allowed_preview_url(url):
        return None
    try:
        response = requests.get(
            url,
            proxies=config.proxy,
            timeout=(10, 30),
            verify=bool(config.app.get("tls_verify", True)),
            headers={"User-Agent": "LocalShortsStudio-GPT-Visual-Review/1"},
        )
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as image:
            image.load()
            return image.convert("RGB")
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
        logger.debug(f"review preview unavailable: {type(exc).__name__}: {exc}")
        return None


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path(utils.font_dir()) / "MicrosoftYaHeiBold.ttc",
        Path(utils.font_dir()) / "Arial.ttf",
        Path("DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    width_chars: int,
    font: ImageFont.ImageFont,
    fill: str = "black",
    line_spacing: int = 5,
) -> int:
    lines = textwrap.wrap(" ".join(str(text or "").split()), width=width_chars) or [""]
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line or " ", font=font)
        y += max(16, bbox[3] - bbox[1]) + line_spacing
    return y


def _candidate_contact_cell(candidate: dict[str, Any], width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    meta_font = _font(17)
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    score = _finite_number(candidate.get("semantic_score"))
    score_text = f"semantic={score:.3f}" if score is not None else "semantic=n/a"
    draw.text((14, 10), f"{candidate_id}   {score_text}", font=title_font, fill="black")

    preview_top = 48
    preview_bottom = height - 56
    frame_width = max(1, (width - 28 - 2 * 8) // CONTACT_SHEET_PREVIEWS)
    frame_height = max(1, preview_bottom - preview_top)
    urls = list(candidate.get("preview_urls") or [])
    if len(urls) > CONTACT_SHEET_PREVIEWS:
        indexes = [
            round(i * (len(urls) - 1) / (CONTACT_SHEET_PREVIEWS - 1))
            for i in range(CONTACT_SHEET_PREVIEWS)
        ]
        urls = [urls[index] for index in indexes]

    for index in range(CONTACT_SHEET_PREVIEWS):
        left = 14 + index * (frame_width + 8)
        box = (left, preview_top, left + frame_width, preview_bottom)
        image = _download_preview(urls[index]) if index < len(urls) else None
        if image is None:
            draw.rectangle(box, outline="#999999", width=2)
            draw.text((left + 10, preview_top + 20), "preview unavailable", font=meta_font, fill="#666666")
            continue
        fitted = ImageOps.fit(image, (frame_width, frame_height), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (left, preview_top))

    provider = str(candidate.get("provider") or "")
    asset = str(candidate.get("asset_id") or "")
    draw.text((14, height - 42), f"{provider}  asset={asset}", font=meta_font, fill="#444444")
    return canvas


def _create_contact_sheet(scene: dict[str, Any], destination: Path) -> None:
    width = 1320
    header_height = 230
    gap = 20
    cell_width = (width - gap * 3) // 2
    cell_height = 430
    height = header_height + gap + 2 * cell_height + gap * 2
    canvas = Image.new("RGB", (width, height), "#f2f2f2")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30)
    text_font = _font(21)
    scene_number = int(scene.get("scene") or 0)
    draw.text(
        (20, 15),
        f"SCENE {scene_number:02d}   {float(scene.get('start') or 0):.1f}s - {float(scene.get('end') or 0):.1f}s",
        font=title_font,
        fill="black",
    )
    y = _draw_wrapped(
        draw,
        f"Narration: {scene.get('narration') or ''}",
        (20, 60),
        width_chars=105,
        font=text_font,
    )
    _draw_wrapped(
        draw,
        f"Visual intent: {scene.get('visual_intent') or ''}",
        (20, y + 4),
        width_chars=105,
        font=text_font,
    )

    candidates = list(scene.get("candidates") or [])[:MAX_REVIEW_CANDIDATES]
    for index in range(MAX_REVIEW_CANDIDATES):
        row, column = divmod(index, 2)
        x = gap + column * (cell_width + gap)
        y = header_height + gap + row * (cell_height + gap)
        if index < len(candidates):
            cell = _candidate_contact_cell(candidates[index], cell_width, cell_height)
        else:
            cell = Image.new("RGB", (cell_width, cell_height), "white")
            ImageDraw.Draw(cell).text((20, 20), "No candidate", font=text_font, fill="#777777")
        canvas.paste(cell, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90, optimize=True)


def _manifest_scene(scene: dict[str, Any], *, included: bool) -> dict[str, Any]:
    return {
        "scene": int(scene.get("scene") or 0),
        "start": float(scene.get("start") or 0),
        "end": float(scene.get("end") or 0),
        "duration": float(scene.get("duration") or 0),
        "narration": str(scene.get("narration") or ""),
        "visual_intent": str(scene.get("visual_intent") or ""),
        "semantic_query": str(scene.get("semantic_query") or ""),
        "current_candidate_id": str(scene.get("current_candidate_id") or ""),
        "review_required": bool(scene.get("review_required")),
        "review_reasons": list(scene.get("review_reasons") or []),
        "included_in_package": bool(included),
        "contact_sheet": f"scene_{int(scene.get('scene') or 0):02d}.jpg" if included else None,
        "candidates": [
            _manifest_candidate(candidate) for candidate in list(scene.get("candidates") or [])
        ] if included else [],
    }


def _read_script_data(task_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(_script_path(task_id).read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualReviewError("task script.json is missing or unreadable") from exc
    if not isinstance(payload, dict):
        raise VisualReviewError("task script.json has an unsupported format")
    return payload


def _review_readme() -> str:
    return """LocalShortsStudio GPT Visual Review Package\n\nThis package is intentionally offline/manual: it does not call the OpenAI API.\nUse the dedicated ChatGPT visual-director chat established for this project.\n\nReview current_video.mp4, review_manifest.json, and every included scene contact sheet.\nReturn decisions ONLY for scenes where included_in_package=true. Scenes with\nincluded_in_package=false are explicitly outside this review round.\n\nAllowed actions: keep, select, retry_search. Never invent candidate IDs. For\nretry_search, describe visible stock-footage content and list misleading visuals in avoid.\nReturn one valid JSON object with schema_version=1 and the exact task_id.\n"""


def export_review_package(task_id: str, scope: str = "recommended") -> dict[str, Any]:
    registry = load_registry(task_id)
    scenes = sorted(registry["scenes"], key=lambda value: int(value.get("scene") or 0))
    if scope not in {"recommended", "all", "retry"}:
        raise VisualReviewError("review scope must be recommended, all, or retry")
    if scope == "all":
        included_numbers = {int(scene.get("scene") or 0) for scene in scenes}
    elif scope == "retry":
        included_numbers = {
            int(scene.get("scene") or 0) for scene in scenes if scene.get("retry_pending")
        }
    else:
        included_numbers = {
            int(scene.get("scene") or 0)
            for scene in scenes
            if scene.get("review_required") or scene.get("retry_pending")
        }
        # A task can be unusually clear according to heuristics.  Still provide a usable
        # package instead of producing an empty ZIP when the user explicitly requests one.
        if not included_numbers:
            included_numbers = {int(scene.get("scene") or 0) for scene in scenes}

    task_dir = _safe_task_directory(task_id)
    review_dir = task_dir / REVIEW_DIR_NAME
    revision = int(registry.get("package_revision") or 0) + 1
    package_dir = review_dir / f"package-v{revision}"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest_scenes = []
    for scene in scenes:
        scene_number = int(scene.get("scene") or 0)
        included = scene_number in included_numbers
        manifest_scenes.append(_manifest_scene(scene, included=included))
        if included:
            _create_contact_sheet(scene, package_dir / f"scene_{scene_number:02d}.jpg")

    current_video = _current_video(task_dir)
    if current_video is not None:
        shutil.copy2(current_video, package_dir / "current_video.mp4")

    script_data = _read_script_data(task_id)
    params = script_data.get("params") if isinstance(script_data.get("params"), dict) else {}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": REVIEW_SCHEMA_VERSION,
        "task_id": str(task_id),
        "package_revision": revision,
        "scope": scope,
        "video_subject": str(params.get("video_subject") or ""),
        "video_duration": max((float(scene.get("end") or 0) for scene in scenes), default=0),
        "review_scene_count": len(included_numbers),
        "total_scene_count": len(scenes),
        "current_video": "current_video.mp4" if current_video is not None else None,
        "scenes": manifest_scenes,
    }
    _write_json_atomic(package_dir / "review_manifest.json", manifest)
    (package_dir / "README_FOR_GPT.txt").write_text(_review_readme(), encoding="utf-8")

    zip_path = review_dir / f"gpt-review-package-v{revision}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(package_dir.iterdir()):
            if file.is_file():
                archive.write(file, arcname=file.name)

    registry["package_revision"] = revision
    registry["last_package"] = {
        "revision": revision,
        "scope": scope,
        "zip_path": str(zip_path),
        "included_scenes": sorted(included_numbers),
    }
    _write_json_atomic(_registry_path(task_id), registry)
    return {
        "task_id": str(task_id),
        "revision": revision,
        "scope": scope,
        "included_scenes": sorted(included_numbers),
        "zip_path": str(zip_path),
        "package_dir": str(package_dir),
    }


def _allowed_fields(action: str) -> set[str]:
    common = {"scene", "action", "confidence", "reason"}
    if action in {"keep", "select"}:
        return common | {"candidate_id"}
    return common | {"search_query", "avoid"}


def validate_decisions(task_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    registry = load_registry(task_id)
    if not isinstance(payload, dict):
        raise VisualReviewError("GPT decisions must be one JSON object")
    if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise VisualReviewError("unsupported GPT decision schema_version")
    if str(payload.get("task_id") or "") != str(task_id):
        raise VisualReviewError("GPT decision task_id does not match the selected task")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise VisualReviewError("GPT decisions must contain a decisions array")

    last_package = registry.get("last_package") if isinstance(registry.get("last_package"), dict) else {}
    expected = {int(value) for value in last_package.get("included_scenes", [])}
    if not expected:
        raise VisualReviewError("build a GPT Review Package before importing decisions")
    scene_map = {int(scene.get("scene") or 0): scene for scene in registry["scenes"]}
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()

    for raw in decisions:
        if not isinstance(raw, dict):
            raise VisualReviewError("every GPT decision must be an object")
        try:
            scene_number = int(raw.get("scene"))
        except (TypeError, ValueError) as exc:
            raise VisualReviewError("every GPT decision needs a numeric scene") from exc
        if scene_number not in expected or scene_number not in scene_map:
            raise VisualReviewError(f"scene {scene_number} was not part of the last review package")
        if scene_number in seen:
            raise VisualReviewError(f"scene {scene_number} appears more than once")
        seen.add(scene_number)
        action = str(raw.get("action") or "").strip().lower()
        if action not in _ALLOWED_ACTIONS:
            raise VisualReviewError(f"scene {scene_number} has unsupported action {action!r}")
        unknown = set(raw) - _allowed_fields(action)
        if unknown:
            raise VisualReviewError(
                f"scene {scene_number} contains unsupported fields: {', '.join(sorted(unknown))}"
            )
        confidence = _finite_number(raw.get("confidence"))
        if confidence is None or confidence < 0 or confidence > 1:
            raise VisualReviewError(f"scene {scene_number} confidence must be between 0 and 1")
        scene = scene_map[scene_number]
        candidate_ids = {
            str(candidate.get("candidate_id") or "") for candidate in scene.get("candidates", [])
        }
        decision: dict[str, Any] = {
            "scene": scene_number,
            "action": action,
            "confidence": confidence,
            "reason": str(raw.get("reason") or "").strip()[:500],
        }
        if action == "keep":
            candidate_id = str(raw.get("candidate_id") or "")
            if candidate_id != str(scene.get("current_candidate_id") or ""):
                raise VisualReviewError(
                    f"scene {scene_number} KEEP must reference its current_candidate_id"
                )
            decision["candidate_id"] = candidate_id
        elif action == "select":
            candidate_id = str(raw.get("candidate_id") or "")
            if not candidate_id or candidate_id not in candidate_ids:
                raise VisualReviewError(f"scene {scene_number} SELECT invented an unknown candidate_id")
            decision["candidate_id"] = candidate_id
        else:
            query = " ".join(str(raw.get("search_query") or "").split())
            if (
                not query
                or len(query) > MAX_RETRY_QUERY_CHARS
                or len(query.split()) > MAX_RETRY_QUERY_WORDS
                or "://" in query
            ):
                raise VisualReviewError(
                    f"scene {scene_number} retry search_query must be a short visible-footage query"
                )
            raw_avoid = raw.get("avoid", [])
            if not isinstance(raw_avoid, list) or len(raw_avoid) > MAX_AVOID_ITEMS:
                raise VisualReviewError(f"scene {scene_number} avoid must be a short list")
            avoid: list[str] = []
            for value in raw_avoid:
                item = " ".join(str(value or "").split())
                if not item or len(item) > MAX_AVOID_CHARS or "://" in item:
                    raise VisualReviewError(f"scene {scene_number} contains an invalid avoid item")
                avoid.append(item)
            decision["search_query"] = query
            decision["avoid"] = avoid
        normalized.append(decision)

    missing = expected - seen
    if missing:
        raise VisualReviewError(
            "GPT decisions omitted reviewed scenes: " + ", ".join(str(value) for value in sorted(missing))
        )
    return normalized


def parse_decision_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(text or ""))
    except json.JSONDecodeError as exc:
        raise VisualReviewError(f"invalid GPT decision JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise VisualReviewError("GPT decisions must be one JSON object")
    return payload


def _find_candidate(scene: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in scene.get("candidates", []):
        if str(candidate.get("candidate_id") or "") == candidate_id:
            return candidate
    raise VisualReviewError(f"candidate {candidate_id!r} is not available for scene {scene.get('scene')}")


def _download_selected_candidate(task_id: str, candidate: dict[str, Any]) -> str:
    existing = str(candidate.get("local_path") or "")
    if existing and Path(existing).is_file():
        return existing
    url = str(candidate.get("download_url") or "")
    if not url.startswith(("http://", "https://")):
        raise VisualReviewError("selected candidate no longer has a downloadable source")
    from app.services import material

    save_dir = _safe_task_directory(task_id) / REVIEW_DIR_NAME / "materials"
    saved = material.save_video(url, str(save_dir))
    if not saved:
        raise VisualReviewError(
            f"failed to download selected candidate {candidate.get('candidate_id')}"
        )
    candidate["local_path"] = saved
    return saved


def _provider_search(
    provider: str,
    query: str,
    minimum_duration: int,
    aspect: VideoAspect,
) -> list[MaterialInfo]:
    from app.services import material

    remote = {
        "pexels": material.search_videos_pexels,
        "pixabay": material.search_videos_pixabay,
        "coverr": material.search_videos_coverr,
    }.get(provider)
    if remote is None:
        raise VisualReviewError(f"manual retry search is not supported for provider {provider!r}")
    return material._search_videos_with_cache(
        provider=provider,
        search_videos=remote,
        search_term=query,
        minimum_duration=minimum_duration,
        video_aspect=aspect,
    )


def _refresh_retry_scene(
    task_id: str,
    scene: dict[str, Any],
    *,
    query: str,
    avoid: list[str],
    params: VideoParams,
) -> None:
    from app.services import semantic_ranker

    provider = str((scene.get("candidates") or [{}])[0].get("provider") or params.video_source)
    raw = _provider_search(
        provider,
        query,
        max(1, int(params.video_clip_duration)),
        VideoAspect(params.video_aspect),
    )
    if not raw:
        raise VisualReviewError(
            f"retry search returned no candidates for scene {scene.get('scene')}"
        )
    ranked = semantic_ranker.rank_materials(
        query,
        raw,
        enabled=True,
        reference_items=[],
        extra_negative_queries=avoid,
    )
    retry_round = int(scene.get("retry_round") or 0) + 1
    candidates = [
        _candidate_payload(
            f"S{int(scene.get('scene') or 0):02d}-R{retry_round}-C{index}",
            item,
        )
        for index, item in enumerate(ranked[:MAX_REVIEW_CANDIDATES], start=1)
    ]
    if not candidates:
        raise VisualReviewError(f"retry ranking produced no candidates for scene {scene.get('scene')}")
    scene["candidates"] = candidates
    # After RETRY_SEARCH there is intentionally no current candidate among the new
    # options.  The next review round must SELECT one; KEEP is therefore invalid until
    # a new current candidate is established.
    scene["current_candidate_id"] = ""
    scene["retry_round"] = retry_round
    scene["retry_pending"] = True
    scene["retry_query"] = query
    scene["retry_avoid"] = avoid
    scene["review_required"] = True
    scene["review_reasons"] = ["gpt_requested_retry_search"]


def _extract_master_audio(source_video: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VisualReviewError("ffmpeg is required to rebuild a reviewed video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source_video),
        "-vn",
        "-c:a",
        "copy",
        str(destination),
    ]
    result = subprocess.run(copy_command, capture_output=True, text=True, check=False)
    if result.returncode == 0 and destination.is_file() and destination.stat().st_size > 0:
        return
    encode_command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source_video),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(destination),
    ]
    result = subprocess.run(encode_command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size <= 0:
        detail = (result.stderr or "ffmpeg audio extraction failed")[-500:]
        raise VisualReviewError(detail)


def _rebuild_reviewed_video(task_id: str, registry: dict[str, Any]) -> str:
    from app.services import video

    task_dir = _safe_task_directory(task_id)
    script_data = _read_script_data(task_id)
    raw_params = script_data.get("params")
    if not isinstance(raw_params, dict):
        raise VisualReviewError("task parameters are unavailable")
    try:
        params = VideoParams.model_validate(raw_params)
    except Exception as exc:
        raise VisualReviewError("task parameters can no longer be validated") from exc

    scenes = sorted(registry["scenes"], key=lambda value: int(value.get("scene") or 0))
    video_paths: list[str] = []
    for scene in scenes:
        current_path = str(scene.get("current_path") or "")
        if not current_path or not Path(current_path).is_file():
            raise VisualReviewError(
                f"scene {scene.get('scene')} has no selected local material; finish its review first"
            )
        video_paths.append(current_path)

    original = task_dir / "final-1.mp4"
    if not original.is_file():
        raise VisualReviewError("the original final-1.mp4 is required for a review rebuild")
    review_dir = task_dir / REVIEW_DIR_NAME
    master_audio = review_dir / "review-master-audio.m4a"
    if not master_audio.is_file() or master_audio.stat().st_size <= 0:
        _extract_master_audio(original, master_audio)

    review_revision = int(registry.get("review_revision") or 0) + 1
    combined = review_dir / f"combined-reviewed-{review_revision}.mp4"
    output = task_dir / f"final-reviewed-{review_revision}.mp4"
    video.combine_videos(
        combined_video_path=str(combined),
        video_paths=video_paths,
        audio_file=str(master_audio),
        video_aspect=params.video_aspect,
        video_fit_mode=params.video_fit_mode,
        video_concat_mode=VideoConcatMode.sequential,
        video_transition_mode=params.video_transition_mode,
        max_clip_duration=params.video_clip_duration,
        threads=params.n_threads,
        clip_speed=params.video_clip_speed,
    )

    review_params = params.model_copy(deep=True)
    # final-1's extracted track already contains the exact narration+BGM mix.  Prevent
    # the renderer from multiplying narration volume again or adding another BGM layer.
    review_params.voice_volume = 1.0
    review_params.bgm_type = ""
    review_params.bgm_volume = 0.0
    review_params.bgm_file = ""
    subtitle_path = task_dir / "subtitle.srt"
    video.generate_video(
        video_path=str(combined),
        audio_path=str(master_audio),
        subtitle_path=str(subtitle_path) if subtitle_path.is_file() else "",
        output_file=str(output),
        params=review_params,
        bgm_file_override="",
    )
    registry["review_revision"] = review_revision
    registry["reviewed_video"] = str(output)
    registry["review_status"] = "complete"
    _write_json_atomic(_registry_path(task_id), registry)
    task_artifacts.patch_script_data(
        task_id,
        gpt_visual_review={
            "available": True,
            "registry_file": REGISTRY_FILE_NAME,
            "scene_count": len(scenes),
            "review_status": "complete",
            "review_revision": review_revision,
            "reviewed_video": output.name,
        },
    )
    return str(output)


def apply_decisions(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    decisions = validate_decisions(task_id, payload)
    registry = load_registry(task_id)
    scene_map = {int(scene.get("scene") or 0): scene for scene in registry["scenes"]}
    params_payload = _read_script_data(task_id).get("params")
    if not isinstance(params_payload, dict):
        raise VisualReviewError("task parameters are unavailable")
    params = VideoParams.model_validate(params_payload)
    retry_scene_numbers: list[int] = []

    for decision in decisions:
        scene = scene_map[decision["scene"]]
        action = decision["action"]
        scene["last_gpt_decision"] = dict(decision)
        if action == "keep":
            scene["review_required"] = False
            scene["review_reasons"] = []
            scene["retry_pending"] = False
            continue
        if action == "select":
            candidate = _find_candidate(scene, decision["candidate_id"])
            selected_path = _download_selected_candidate(task_id, candidate)
            scene["current_candidate_id"] = decision["candidate_id"]
            scene["current_path"] = selected_path
            scene["review_required"] = False
            scene["review_reasons"] = []
            scene["retry_pending"] = False
            continue
        _refresh_retry_scene(
            task_id,
            scene,
            query=decision["search_query"],
            avoid=decision.get("avoid", []),
            params=params,
        )
        retry_scene_numbers.append(decision["scene"])

    registry["scenes"] = sorted(scene_map.values(), key=lambda value: int(value.get("scene") or 0))
    registry["review_status"] = "retry_required" if retry_scene_numbers else "approved"
    _write_json_atomic(_registry_path(task_id), registry)

    if retry_scene_numbers:
        package = export_review_package(task_id, scope="retry")
        return {
            "status": "retry_required",
            "retry_scenes": sorted(retry_scene_numbers),
            "package_zip": package["zip_path"],
            "package_revision": package["revision"],
        }

    reviewed_video = _rebuild_reviewed_video(task_id, registry)
    return {
        "status": "rebuilt",
        "reviewed_video": reviewed_video,
        "review_revision": int(load_registry(task_id).get("review_revision") or 0),
    }


def review_summary(task_id: str) -> dict[str, Any]:
    registry = load_registry(task_id)
    scenes = registry["scenes"]
    task_dir = _safe_task_directory(task_id)
    last_package = registry.get("last_package") if isinstance(registry.get("last_package"), dict) else {}
    zip_path = str(last_package.get("zip_path") or "")
    current_video = _current_video(task_dir)
    return {
        "task_id": str(task_id),
        "scene_count": len(scenes),
        "review_required_count": sum(bool(scene.get("review_required")) for scene in scenes),
        "retry_pending_count": sum(bool(scene.get("retry_pending")) for scene in scenes),
        "package_revision": int(registry.get("package_revision") or 0),
        "review_revision": int(registry.get("review_revision") or 0),
        "last_package_zip": zip_path if zip_path and Path(zip_path).is_file() else "",
        "current_video": str(current_video) if current_video is not None else "",
        "reviewed_video": str(registry.get("reviewed_video") or ""),
        "review_status": str(registry.get("review_status") or "pending"),
    }


def list_review_tasks(limit: int = 50) -> list[dict[str, Any]]:
    tasks_root = Path(utils.task_dir())
    if not tasks_root.is_dir():
        return []
    entries: list[tuple[float, str]] = []
    for entry in tasks_root.iterdir():
        try:
            registry = entry / REGISTRY_FILE_NAME
            if entry.is_dir() and registry.is_file():
                entries.append((registry.stat().st_mtime, entry.name))
        except OSError:
            continue
    entries.sort(reverse=True)
    output = []
    for _, task_id in entries[: max(1, int(limit))]:
        try:
            summary = review_summary(task_id)
            script = _read_script_data(task_id)
            params = script.get("params") if isinstance(script.get("params"), dict) else {}
            summary["subject"] = str(params.get("video_subject") or script.get("script") or task_id)[:120]
            output.append(summary)
        except VisualReviewError:
            continue
    return output
