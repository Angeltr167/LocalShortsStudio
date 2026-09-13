"""Deterministic 2D cartoon renderer for LocalShortsStudio.

The engine follows a deliberately hybrid architecture:

* the configured LLM may act as a *scene director* and choose from a small,
  validated animation vocabulary;
* the actual artwork, character identity, camera layouts, lip sync and render are
  deterministic local code;
* no text-to-video model is required and no external image/video API is called.

This keeps characters visually stable from frame to frame while still letting AI
make semantic decisions about which visual explanation fits each narration beat.
The visual language is original and procedural (rounded doodle characters,
podcast desk, microphones and explanatory UI cards); it does not copy assets from
any reference video or third-party cartoon package.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from app.config import config
from app.models.schema import VideoAspect, VideoParams
from app.services import llm, task_artifacts
from app.utils import utils


class CartoonRenderError(RuntimeError):
    """Raised when the local cartoon renderer cannot produce a valid material."""


CARTOON_PLAN_SCHEMA = "localshortsstudio.ai-cartoon-plan"
CARTOON_PLAN_VERSION = 1
DEFAULT_FPS = 24
MIN_SCENE_SECONDS = 1.6
MAX_SCENE_SECONDS = 6.5
TRANSITION_SECONDS = 0.22
MAX_OVERLAY_TEXT = 42

_LAYOUTS = frozenset({"host", "guest", "two_shot", "graphic"})
_SPEAKERS = frozenset({"host", "guest"})
_EMOTIONS = frozenset({"neutral", "thinking", "surprised", "happy", "concerned"})
_ACTIONS = frozenset({"talk", "explain", "point", "think", "react", "write"})
_OVERLAYS = frozenset(
    {
        "none",
        "stat",
        "browser_tabs",
        "email",
        "checklist",
        "brain",
        "phone",
        "chart",
        "money",
        "clock",
        "comparison",
        "thought",
    }
)
_MOUTH_SHAPES = frozenset("XABCDEFGH")


@dataclass(frozen=True)
class NarrationCue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class MouthCue:
    start: float
    end: float
    value: str


@dataclass
class CartoonScene:
    scene: int
    start: float
    end: float
    narration: str
    layout: str
    speaker: str
    emotion: str
    action: str
    overlay: str
    overlay_text: str = ""
    accent_text: str = ""

    @property
    def duration(self) -> float:
        return max(0.01, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "narration": self.narration,
            "layout": self.layout,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "action": self.action,
            "overlay": self.overlay,
            "overlay_text": self.overlay_text,
            "accent_text": self.accent_text,
        }


@dataclass(frozen=True)
class CartoonPalette:
    background: str = "#11131D"
    background_alt: str = "#1A1D2A"
    grid: str = "#25293A"
    ink: str = "#11131A"
    text: str = "#F7F4EE"
    muted_text: str = "#B9B7C6"
    host: str = "#FF6F91"
    host_shadow: str = "#D44E72"
    guest: str = "#8A7CFF"
    guest_shadow: str = "#685ADA"
    accent: str = "#FFD166"
    cyan: str = "#66D9EF"
    green: str = "#7BD389"
    red: str = "#FF6B6B"
    table: str = "#3A2834"
    table_edge: str = "#241A22"
    card: str = "#F5F0E8"
    card_ink: str = "#171821"


PALETTE = CartoonPalette()


def _parse_srt_timestamp(value: str) -> float:
    value = value.strip().replace(".", ",")
    match = re.fullmatch(r"(\d+):(\d+):(\d+),(\d+)", value)
    if not match:
        raise ValueError(f"invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_srt_cues(path: str | Path) -> list[NarrationCue]:
    subtitle_path = Path(path)
    if not subtitle_path.is_file():
        return []
    text = subtitle_path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    cues: list[NarrationCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timeline_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timeline_index < 0:
            continue
        try:
            left, right = [part.strip() for part in lines[timeline_index].split("-->", 1)]
            start = _parse_srt_timestamp(left)
            end = _parse_srt_timestamp(right)
        except (ValueError, TypeError):
            continue
        cue_text = " ".join(lines[timeline_index + 1 :]).strip()
        if cue_text and end > start:
            cues.append(NarrationCue(start=start, end=end, text=cue_text))
    return cues


def _script_sentences(script: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(script or "")).strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _fallback_timed_cues(script: str, duration: float) -> list[NarrationCue]:
    sentences = _script_sentences(script)
    if not sentences:
        return [NarrationCue(0.0, max(1.0, duration), "")]
    weights = [max(3, len(re.findall(r"\w+", sentence))) for sentence in sentences]
    weight_total = max(1, sum(weights))
    cursor = 0.0
    cues: list[NarrationCue] = []
    for index, (sentence, weight) in enumerate(zip(sentences, weights)):
        if index == len(sentences) - 1:
            end = duration
        else:
            end = min(duration, cursor + duration * weight / weight_total)
        end = max(cursor + 0.25, end)
        cues.append(NarrationCue(cursor, min(duration, end), sentence))
        cursor = min(duration, end)
    return cues


def _merge_tiny_cues(cues: Iterable[NarrationCue], duration: float) -> list[NarrationCue]:
    source = [cue for cue in cues if cue.end > cue.start]
    if not source:
        return []
    merged: list[NarrationCue] = []
    buffer_start = source[0].start
    buffer_end = source[0].end
    buffer_text = [source[0].text]
    for cue in source[1:]:
        buffered_duration = buffer_end - buffer_start
        if buffered_duration < MIN_SCENE_SECONDS or cue.end - buffer_start <= MAX_SCENE_SECONDS:
            buffer_end = cue.end
            buffer_text.append(cue.text)
            # A natural sentence boundary and enough duration are good places to cut.
            if (
                buffer_end - buffer_start >= 2.5
                and re.search(r"[.!?。！？]$", cue.text.strip())
            ):
                merged.append(
                    NarrationCue(buffer_start, buffer_end, " ".join(buffer_text).strip())
                )
                buffer_start = buffer_end
                buffer_text = []
            continue
        merged.append(NarrationCue(buffer_start, buffer_end, " ".join(buffer_text).strip()))
        buffer_start = cue.start
        buffer_end = cue.end
        buffer_text = [cue.text]
    if buffer_text:
        merged.append(NarrationCue(buffer_start, buffer_end, " ".join(buffer_text).strip()))

    # Guarantee visual coverage from t=0 to the narration tail. Small gaps inherit the
    # surrounding scene instead of producing a black frame.
    normalized: list[NarrationCue] = []
    cursor = 0.0
    for cue in merged:
        start = cursor
        end = max(start + 0.15, cue.end)
        normalized.append(NarrationCue(start, min(duration, end), cue.text))
        cursor = min(duration, end)
    if normalized and normalized[-1].end < duration:
        last = normalized[-1]
        normalized[-1] = NarrationCue(last.start, duration, last.text)
    return normalized


def build_narration_timeline(
    script: str,
    audio_duration: float,
    subtitle_path: str = "",
) -> list[NarrationCue]:
    duration = max(float(audio_duration or 0), 0.5)
    cues = parse_srt_cues(subtitle_path) if subtitle_path else []
    if not cues:
        cues = _fallback_timed_cues(script, duration)
    return _merge_tiny_cues(cues, duration) or [NarrationCue(0.0, duration, script)]


def _clean_overlay_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"[`*_#]", "", text).strip()
    return text[:MAX_OVERLAY_TEXT]


def _extract_number(text: str) -> str:
    match = re.search(
        r"(?:[$€£]\s*)?\d[\d,.]*(?:\s*%|\s*(?:dollars?|hours?|minutes?|days?|years?))?",
        text,
        re.IGNORECASE,
    )
    return _clean_overlay_text(match.group(0)) if match else ""


def _short_keyword(text: str, fallback: str = "FOCUS") -> str:
    words = re.findall(r"[A-Za-z0-9$%]+", text)
    ignored = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "your",
        "you",
        "that",
        "this",
        "with",
        "from",
        "into",
        "for",
        "are",
        "is",
        "was",
        "have",
        "has",
        "can",
    }
    useful = [word for word in words if word.lower() not in ignored]
    chosen = useful[:4] or words[:4]
    return " ".join(chosen).upper()[:MAX_OVERLAY_TEXT] or fallback


def _fallback_overlay(text: str) -> tuple[str, str]:
    lower = text.lower()
    number = _extract_number(text)
    if number:
        if any(token in lower for token in ("money", "dollar", "cost", "price", "$")):
            return "money", number
        return "stat", number
    if any(token in lower for token in ("browser", "tab", "tabs", "website")):
        return "browser_tabs", "UNFINISHED TABS"
    if any(token in lower for token in ("email", "inbox", "message", "draft")):
        return "email", "DRAFT"
    if any(token in lower for token in ("checklist", "to-do", "todo", "write down", "next step")):
        return "checklist", "NEXT STEP"
    if any(token in lower for token in ("brain", "memory", "remember", "attention", "focus", "mind")):
        return "brain", _short_keyword(text, "ATTENTION")
    if any(token in lower for token in ("phone", "notification", "social media", "smartphone")):
        return "phone", "NOTIFICATIONS"
    if any(token in lower for token in ("time", "minute", "hour", "morning", "night")):
        return "clock", _short_keyword(text, "TIME")
    if any(token in lower for token in ("compare", "versus", "instead", "rather than", "difference")):
        return "comparison", _short_keyword(text, "A vs B")
    if any(token in lower for token in ("increase", "decrease", "percent", "data", "research", "study")):
        return "chart", _short_keyword(text, "TREND")
    if any(token in lower for token in ("think", "idea", "imagine", "reason", "why")):
        return "thought", _short_keyword(text, "IDEA")
    return "none", ""


def _fallback_scene(cue: NarrationCue, index: int) -> CartoonScene:
    overlay, overlay_text = _fallback_overlay(cue.text)
    if overlay != "none" and index % 3 == 2:
        layout = "graphic"
    else:
        layout = ("host", "guest", "two_shot")[index % 3]
    speaker = "host" if index % 2 == 0 else "guest"
    lower = cue.text.lower()
    if any(token in lower for token in ("problem", "stress", "ruin", "forget", "unfinished", "warning")):
        emotion = "concerned"
    elif any(token in lower for token in ("surpris", "suddenly", "actually", "imagine")):
        emotion = "surprised"
    elif any(token in lower for token in ("try", "use this", "advantage", "simple", "better")):
        emotion = "happy"
    elif any(token in lower for token in ("think", "brain", "reason", "memory", "mind")):
        emotion = "thinking"
    else:
        emotion = "neutral"
    if overlay in {"checklist", "email"}:
        action = "write"
    elif overlay != "none":
        action = "point"
    elif emotion == "thinking":
        action = "think"
    else:
        action = "explain" if index % 2 == 0 else "talk"
    return CartoonScene(
        scene=index + 1,
        start=cue.start,
        end=cue.end,
        narration=cue.text,
        layout=layout,
        speaker=speaker,
        emotion=emotion,
        action=action,
        overlay=overlay,
        overlay_text=overlay_text,
        accent_text=_short_keyword(cue.text, ""),
    )


def _json_payload_from_response(text: str) -> Any:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _sanitize_scene_choice(raw: dict[str, Any], fallback: CartoonScene) -> CartoonScene:
    def choice(key: str, allowed: frozenset[str], default: str) -> str:
        value = str(raw.get(key) or "").strip().lower()
        return value if value in allowed else default

    overlay = choice("overlay", _OVERLAYS, fallback.overlay)
    overlay_text = _clean_overlay_text(raw.get("overlay_text"))
    if overlay != "none" and not overlay_text:
        overlay_text = fallback.overlay_text or _short_keyword(fallback.narration)
    return CartoonScene(
        scene=fallback.scene,
        start=fallback.start,
        end=fallback.end,
        narration=fallback.narration,
        layout=choice("layout", _LAYOUTS, fallback.layout),
        speaker=choice("speaker", _SPEAKERS, fallback.speaker),
        emotion=choice("emotion", _EMOTIONS, fallback.emotion),
        action=choice("action", _ACTIONS, fallback.action),
        overlay=overlay,
        overlay_text=overlay_text,
        accent_text=_clean_overlay_text(raw.get("accent_text")) or fallback.accent_text,
    )


def _director_prompt(subject: str, fallbacks: list[CartoonScene]) -> str:
    compact_scenes = [
        {
            "scene": scene.scene,
            "start": round(scene.start, 2),
            "end": round(scene.end, 2),
            "narration": scene.narration,
            "fallback_overlay": scene.overlay,
        }
        for scene in fallbacks
    ]
    return f"""
You are a visual director for an original 2D doodle podcast explainer.
Choose a clear visual treatment for every narration scene. The actual renderer is
procedural, so you MUST use only the allowed enum values below.

Video subject: {subject}

Allowed layout: host, guest, two_shot, graphic
Allowed speaker: host, guest
Allowed emotion: neutral, thinking, surprised, happy, concerned
Allowed action: talk, explain, point, think, react, write
Allowed overlay: none, stat, browser_tabs, email, checklist, brain, phone, chart,
money, clock, comparison, thought

Rules:
- preserve scene numbers and return exactly one entry per supplied scene;
- choose visuals that communicate meaning, not just object overlap;
- use graphic overlays when a concept, number, UI state or comparison benefits from it;
- overlay_text must be visible on screen, concise (ideally 1-5 words), and never exceed 42 characters;
- alternate host/guest/two-shot enough to avoid visual monotony;
- do not request stock footage, images, camera realism, brands, copyrighted characters, or new enum values;
- return ONLY valid JSON, no markdown and no explanation.

Output shape:
{{"scenes":[{{"scene":1,"layout":"host","speaker":"host","emotion":"neutral","action":"explain","overlay":"none","overlay_text":"","accent_text":""}}]}}

Scenes:
{json.dumps(compact_scenes, ensure_ascii=False)}
""".strip()


def build_cartoon_plan(
    *,
    subject: str,
    script: str,
    audio_duration: float,
    subtitle_path: str = "",
    ai_director: bool = True,
    app_config=None,
) -> tuple[list[CartoonScene], str]:
    timeline = build_narration_timeline(script, audio_duration, subtitle_path)
    fallbacks = [_fallback_scene(cue, index) for index, cue in enumerate(timeline)]
    if not ai_director:
        return fallbacks, "deterministic"

    response = llm.generate_cartoon_direction(
        prompt=_director_prompt(subject, fallbacks),
        app_config=app_config,
    )
    if not response or response.startswith("Error:"):
        logger.warning(f"AI cartoon director unavailable; using deterministic plan: {response}")
        return fallbacks, "deterministic_fallback"
    try:
        payload = _json_payload_from_response(response)
        raw_scenes = payload.get("scenes") if isinstance(payload, dict) else payload
        if not isinstance(raw_scenes, list):
            raise ValueError("AI cartoon plan must contain a scenes list")
        by_number = {
            int(item.get("scene")): item
            for item in raw_scenes
            if isinstance(item, dict) and str(item.get("scene", "")).isdigit()
        }
        directed = [
            _sanitize_scene_choice(by_number.get(fallback.scene, {}), fallback)
            for fallback in fallbacks
        ]
        return directed, "ai"
    except Exception as exc:
        logger.warning(f"invalid AI cartoon plan; using deterministic plan: {exc}")
        return fallbacks, "deterministic_fallback"


def _resolve_rhubarb_binary() -> str:
    configured = str(config.app.get("rhubarb_path", "") or "").strip()
    candidates = [
        configured,
        os.environ.get("RHUBARB_PATH", ""),
        shutil.which("rhubarb") or "",
        str(Path(utils.root_dir()) / "tools" / "rhubarb" / "rhubarb.exe"),
        str(Path(utils.root_dir()) / "tools" / "rhubarb" / "rhubarb"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return ""


def _parse_rhubarb_payload(payload: dict[str, Any]) -> list[MouthCue]:
    result: list[MouthCue] = []
    raw = payload.get("mouthCues") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return result
    for cue in raw:
        if not isinstance(cue, dict):
            continue
        try:
            start = float(cue.get("start"))
            end = float(cue.get("end"))
        except (TypeError, ValueError):
            continue
        value = str(cue.get("value") or "X").upper()
        if value not in _MOUTH_SHAPES:
            value = "X"
        if end > start:
            result.append(MouthCue(start, end, value))
    return result


def _heuristic_audio_mouth_cues(audio_file: str) -> list[MouthCue]:
    """Build deterministic mouth activity from local narration amplitude.

    This is intentionally not a phoneme recognizer. It prevents the fallback mouth
    cycle from flapping through silence while keeping a lively mix of mouth shapes.
    Rhubarb remains the higher-precision optional path.
    """
    ffmpeg = utils.get_ffmpeg_binary()
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(audio_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"heuristic lip-sync audio analysis unavailable: {exc}")
        return []
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace")[-300:]
        logger.warning(
            "heuristic lip-sync audio analysis failed"
            + (f": {detail}" if detail else "")
        )
        return []

    samples = array("h")
    samples.frombytes(completed.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return []

    sample_rate = 16000
    window_samples = int(sample_rate * 0.09)
    energies: list[float] = []
    bounds: list[tuple[int, int]] = []
    for start in range(0, len(samples), window_samples):
        end = min(len(samples), start + window_samples)
        if end <= start:
            continue
        total = 0.0
        for sample in samples[start:end]:
            total += float(sample) * float(sample)
        rms = math.sqrt(total / (end - start)) / 32768.0
        energies.append(rms)
        bounds.append((start, end))
    if not energies:
        return []

    ordered = sorted(energies)

    def percentile(fraction: float) -> float:
        index = int(round((len(ordered) - 1) * fraction))
        return ordered[max(0, min(len(ordered) - 1, index))]

    noise_floor = percentile(0.20)
    speech_level = max(percentile(0.82), noise_floor + 1e-6)
    threshold = max(0.0055, noise_floor * 1.9, speech_level * 0.16)
    shape_cycle = "ACBEDFGH"
    cues: list[MouthCue] = []
    for index, (energy, (start, end)) in enumerate(zip(energies, bounds)):
        start_t = start / sample_rate
        end_t = end / sample_rate
        if energy <= threshold:
            shape = "X"
        else:
            strength = min(
                1.0,
                max(0.0, (energy - threshold) / max(1e-6, speech_level - threshold)),
            )
            offset = 2 if strength > 0.72 else (1 if strength > 0.38 else 0)
            shape = shape_cycle[(index + offset) % len(shape_cycle)]
        cues.append(MouthCue(start_t, end_t, shape))
    return cues


def generate_mouth_cues(
    audio_file: str,
    mode: Literal["auto", "rhubarb", "heuristic"] = "auto",
) -> tuple[list[MouthCue], str]:
    if mode == "heuristic":
        return _heuristic_audio_mouth_cues(audio_file), "heuristic"
    binary = _resolve_rhubarb_binary()
    if not binary:
        if mode == "rhubarb":
            raise CartoonRenderError(
                "Rhubarb lip sync was requested but no rhubarb binary was found. "
                "Set app.rhubarb_path, RHUBARB_PATH, or place it under tools/rhubarb/."
            )
        return _heuristic_audio_mouth_cues(audio_file), "heuristic"
    with tempfile.TemporaryDirectory(prefix="mpt-rhubarb-") as temp_dir:
        output = Path(temp_dir) / "mouth.json"
        command = [binary, "-f", "json", "-o", str(output), str(audio_file)]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            if mode == "rhubarb":
                raise CartoonRenderError(f"Rhubarb lip sync failed: {exc}") from exc
            logger.warning(f"Rhubarb unavailable; falling back to heuristic lip sync: {exc}")
            return _heuristic_audio_mouth_cues(audio_file), "heuristic"
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout or "rhubarb failed")[-500:]
            if mode == "rhubarb":
                raise CartoonRenderError(detail)
            logger.warning(f"Rhubarb failed; falling back to heuristic lip sync: {detail}")
            return _heuristic_audio_mouth_cues(audio_file), "heuristic"
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if mode == "rhubarb":
                raise CartoonRenderError(f"invalid Rhubarb output: {exc}") from exc
            return _heuristic_audio_mouth_cues(audio_file), "heuristic"
        cues = _parse_rhubarb_payload(payload)
        if cues:
            return cues, "rhubarb"
        return _heuristic_audio_mouth_cues(audio_file), "heuristic"


def _mouth_at(cues: list[MouthCue], t: float, speaking: bool) -> str:
    if not speaking:
        return "X"
    for cue in cues:
        if cue.start <= t < cue.end:
            return cue.value
    if cues:
        # Audio-derived cues explicitly encode quiet windows. When no cue covers this
        # timestamp, close the mouth instead of reviving the time-only fallback cycle.
        return "X"
    # Fast deterministic fallback. It is intentionally irregular rather than a simple
    # open/closed toggle, which makes narration feel substantially less robotic.
    sequence = "ACDEBFAGDC"
    return sequence[int(max(t, 0) / 0.105) % len(sequence)]


def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    candidates = [
        Path(utils.font_dir()) / "MicrosoftYaHeiBold.ttc",
        Path(utils.font_dir()) / "Arial.ttf",
        Path("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=max(8, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def _ease_out_cubic(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 1.0 - (1.0 - value) ** 3


def _lerp(a: float, b: float, amount: float) -> float:
    return a + (b - a) * amount


def _scene_for_time(scenes: list[CartoonScene], t: float) -> CartoonScene:
    for scene in scenes:
        if scene.start <= t < scene.end:
            return scene
    return scenes[-1]


def _text_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1]), text, font=font, fill=fill)


def _wrap_lines(text: str, width: int = 22, max_lines: int = 3) -> list[str]:
    lines = textwrap.wrap(" ".join(str(text or "").split()), width=width)
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    kept[-1] = kept[-1].rstrip(" .") + "…"
    return kept


class DoodleRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        scenes: list[CartoonScene],
        mouth_cues: list[MouthCue],
    ) -> None:
        self.width = width
        self.height = height
        self.scenes = scenes
        self.mouth_cues = mouth_cues
        self.scale = min(width / 1080.0, height / 1920.0)
        self.palette = PALETTE

    def sx(self, value: float) -> int:
        return int(round(value * self.width / 1080.0))

    def sy(self, value: float) -> int:
        return int(round(value * self.height / 1920.0))

    def sc(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def _background(self, draw: ImageDraw.ImageDraw, t: float) -> None:
        w, h = self.width, self.height
        draw.rectangle((0, 0, w, h), fill=self.palette.background)
        band_h = max(1, h // 14)
        for index in range(14):
            shade = 18 + index
            draw.rectangle(
                (0, index * band_h, w, (index + 1) * band_h),
                fill=(shade, shade + 2, shade + 13),
            )
        # Original studio wall: soft acoustic capsules + small indicator lamps.
        for col in range(5):
            x = self.sx(90 + col * 225)
            y = self.sy(155 + (col % 2) * 55)
            draw.rounded_rectangle(
                (x, y, x + self.sx(125), y + self.sy(330)),
                radius=self.sc(50),
                fill=self.palette.background_alt,
                outline=self.palette.grid,
                width=self.sc(4),
            )
        pulse = 0.5 + 0.5 * math.sin(t * 1.4)
        lamp = int(160 + 70 * pulse)
        draw.ellipse(
            (self.sx(905), self.sy(120), self.sx(930), self.sy(145)),
            fill=(lamp, 90, 115),
        )
        # Floor shadow behind the table.
        draw.ellipse(
            (self.sx(70), self.sy(1460), self.sx(1010), self.sy(1770)),
            fill="#0D0F17",
        )

    def _table(self, draw: ImageDraw.ImageDraw) -> None:
        top = self.sy(1315)
        draw.rounded_rectangle(
            (self.sx(45), top, self.sx(1035), self.sy(1800)),
            radius=self.sc(70),
            fill=self.palette.table,
            outline=self.palette.table_edge,
            width=self.sc(9),
        )
        draw.rectangle(
            (self.sx(55), top, self.sx(1025), top + self.sy(26)),
            fill="#704256",
        )

    def _microphone(self, draw: ImageDraw.ImageDraw, x: int, y: int, flip: bool) -> None:
        ink = "#08090D"
        direction = -1 if flip else 1
        draw.line(
            (x, y + self.sy(80), x + direction * self.sx(70), y + self.sy(10)),
            fill=ink,
            width=self.sc(13),
        )
        draw.line(
            (x, y + self.sy(80), x, y + self.sy(190)),
            fill=ink,
            width=self.sc(12),
        )
        capsule_x = x + direction * self.sx(74)
        draw.rounded_rectangle(
            (
                capsule_x - self.sx(34),
                y - self.sy(35),
                capsule_x + self.sx(34),
                y + self.sy(72),
            ),
            radius=self.sc(25),
            fill="#2A2E3B",
            outline="#07080B",
            width=self.sc(7),
        )
        for offset in (-17, 0, 17):
            draw.line(
                (
                    capsule_x - self.sx(21),
                    y + self.sy(offset),
                    capsule_x + self.sx(21),
                    y + self.sy(offset),
                ),
                fill="#5E6474",
                width=self.sc(3),
            )

    def _mouth(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        shape: str,
        scale: float,
    ) -> None:
        ink = self.palette.ink
        sw = max(2, self.sc(6 * scale))
        width = self.sx(58 * scale)
        height = self.sy(30 * scale)
        if shape == "X":
            draw.arc((x - width, y - height / 2, x + width, y + height), 10, 170, fill=ink, width=sw)
            return
        if shape in {"A", "B"}:
            open_h = self.sy((18 if shape == "A" else 30) * scale)
            draw.ellipse((x - width * 0.72, y - open_h, x + width * 0.72, y + open_h), fill=ink)
            return
        if shape in {"C", "D"}:
            open_h = self.sy((25 if shape == "C" else 38) * scale)
            draw.rounded_rectangle(
                (x - width * 0.82, y - open_h, x + width * 0.82, y + open_h),
                radius=max(2, self.sc(12 * scale)),
                fill=ink,
            )
            tongue_y = y + open_h * 0.25
            draw.ellipse(
                (x - width * 0.45, tongue_y, x + width * 0.45, y + open_h * 0.9),
                fill="#FF8BA7",
            )
            return
        if shape in {"E", "F"}:
            draw.ellipse(
                (x - width * 0.45, y - height * 0.7, x + width * 0.45, y + height * 1.1),
                fill=ink,
            )
            return
        # G/H: smile-ish open mouth.
        draw.arc((x - width, y - height, x + width, y + height * 1.7), 5, 175, fill=ink, width=sw)
        draw.line((x - width * 0.75, y, x + width * 0.75, y), fill=ink, width=sw)

    def _character(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        x: int,
        base_y: int,
        color: str,
        shadow: str,
        emotion: str,
        action: str,
        mouth: str,
        facing: int,
        t: float,
        local_progress: float,
        scale: float = 1.0,
        dimmed: bool = False,
    ) -> None:
        identity_host = color == self.palette.host
        phase = 0.0 if identity_host else 0.85
        bob = self.sy(math.sin(t * 3.4 + phase) * 6)
        entrance = _ease_out_cubic(min(1.0, local_progress * 8.0))
        nod = self.sy(math.sin(t * 2.1 + phase) * 2.5)
        y = base_y + bob + nod + self.sy((1.0 - entrance) * 28)
        outline = self.palette.ink
        alpha_color = color
        if dimmed:
            # Keep both people present in two-shots but visually prioritize the speaker.
            alpha_color = shadow

        body_w = self.sx(230 * scale)
        body_h = self.sy(330 * scale)
        head_r = self.sx(125 * scale)
        draw.rounded_rectangle(
            (x - body_w / 2, y, x + body_w / 2, y + body_h),
            radius=max(self.sc(28), self.sc(82 * scale)),
            fill=alpha_color,
            outline=outline,
            width=self.sc(9 * scale),
        )
        # Stable clothing details make the recurring host/guest read as characters
        # rather than interchangeable colored blobs.
        collar_y = y + self.sy(48 * scale)
        draw.polygon(
            (
                (x - self.sx(58 * scale), collar_y),
                (x, collar_y + self.sy(52 * scale)),
                (x + self.sx(58 * scale), collar_y),
                (x, collar_y + self.sy(108 * scale)),
            ),
            fill=shadow,
        )
        badge_x = x - self.sx(62 * scale) if identity_host else x + self.sx(62 * scale)
        draw.ellipse(
            (
                badge_x - self.sx(14 * scale),
                y + self.sy(155 * scale),
                badge_x + self.sx(14 * scale),
                y + self.sy(183 * scale),
            ),
            fill=self.palette.accent if identity_host else self.palette.cyan,
            outline=outline,
            width=max(1, self.sc(3 * scale)),
        )
        head_y = y - self.sy(105 * scale)
        # Ears sit behind the face and help the head feel constructed rather than stamped.
        ear_r = self.sx(23 * scale)
        for ear_x in (x - head_r, x + head_r):
            draw.ellipse(
                (
                    ear_x - ear_r,
                    head_y - ear_r * 0.15,
                    ear_x + ear_r,
                    head_y + ear_r * 1.85,
                ),
                fill=shadow,
                outline=outline,
                width=max(1, self.sc(5 * scale)),
            )
        draw.ellipse(
            (x - head_r, head_y - head_r, x + head_r, head_y + head_r),
            fill=color,
            outline=outline,
            width=self.sc(9 * scale),
        )
        # Distinct silhouettes: the host has a three-point quiff, the guest a side sweep.
        hair_fill = "#342A38" if identity_host else "#302E55"
        if identity_host:
            draw.polygon(
                (
                    (x - self.sx(72 * scale), head_y - self.sy(102 * scale)),
                    (x - self.sx(28 * scale), head_y - self.sy(145 * scale)),
                    (x + self.sx(2 * scale), head_y - self.sy(106 * scale)),
                    (x + self.sx(42 * scale), head_y - self.sy(142 * scale)),
                    (x + self.sx(78 * scale), head_y - self.sy(98 * scale)),
                ),
                fill=hair_fill,
            )
        else:
            draw.pieslice(
                (
                    x - self.sx(108 * scale),
                    head_y - self.sy(128 * scale),
                    x + self.sx(108 * scale),
                    head_y + self.sy(18 * scale),
                ),
                188,
                352,
                fill=hair_fill,
            )
            draw.ellipse(
                (
                    x + self.sx(68 * scale),
                    head_y - self.sy(62 * scale),
                    x + self.sx(113 * scale),
                    head_y + self.sy(22 * scale),
                ),
                fill=hair_fill,
            )
        # Small asymmetric cheek patch gives the characters an authored, non-template feel.
        draw.ellipse(
            (
                x + facing * self.sx(58 * scale) - self.sx(18 * scale),
                head_y + self.sy(28 * scale),
                x + facing * self.sx(58 * scale) + self.sx(18 * scale),
                head_y + self.sy(52 * scale),
            ),
            fill=shadow,
        )

        blink_phase = (t + (0.7 if facing < 0 else 0.0)) % 4.1
        blink = 3.82 < blink_phase < 3.96
        eye_y = head_y - self.sy(20 * scale)
        eye_gap = self.sx(48 * scale)
        eye_r = self.sc(9 * scale)
        if blink:
            for eye_x in (x - eye_gap, x + eye_gap):
                draw.line(
                    (eye_x - eye_r, eye_y, eye_x + eye_r, eye_y),
                    fill=outline,
                    width=self.sc(5 * scale),
                )
        else:
            pupil_shift = facing * self.sc(2 * scale)
            for eye_x in (x - eye_gap, x + eye_gap):
                white_r = self.sc(13 * scale)
                draw.ellipse(
                    (
                        eye_x - white_r,
                        eye_y - white_r,
                        eye_x + white_r,
                        eye_y + white_r,
                    ),
                    fill="#F8F5F0",
                    outline=outline,
                    width=max(1, self.sc(3 * scale)),
                )
                pupil_r = self.sc(6 * scale)
                draw.ellipse(
                    (
                        eye_x + pupil_shift - pupil_r,
                        eye_y - pupil_r,
                        eye_x + pupil_shift + pupil_r,
                        eye_y + pupil_r,
                    ),
                    fill=outline,
                )

        brow_y = eye_y - self.sy(30 * scale)
        if emotion == "concerned":
            draw.line((x - eye_gap - eye_r, brow_y - eye_r, x - eye_gap + eye_r, brow_y), fill=outline, width=self.sc(5 * scale))
            draw.line((x + eye_gap - eye_r, brow_y, x + eye_gap + eye_r, brow_y - eye_r), fill=outline, width=self.sc(5 * scale))
        elif emotion == "surprised":
            for eye_x in (x - eye_gap, x + eye_gap):
                draw.arc((eye_x - self.sx(18 * scale), brow_y - self.sy(18 * scale), eye_x + self.sx(18 * scale), brow_y + self.sy(18 * scale)), 190, 350, fill=outline, width=self.sc(5 * scale))
        elif emotion == "thinking":
            draw.line((x - eye_gap - eye_r, brow_y, x - eye_gap + eye_r, brow_y - eye_r), fill=outline, width=self.sc(5 * scale))
            draw.line((x + eye_gap - eye_r, brow_y - eye_r, x + eye_gap + eye_r, brow_y), fill=outline, width=self.sc(5 * scale))
        else:
            for eye_x in (x - eye_gap, x + eye_gap):
                draw.line((eye_x - eye_r, brow_y, eye_x + eye_r, brow_y), fill=outline, width=self.sc(4 * scale))

        # Tiny nose mark keeps expressions readable at vertical-video scale.
        nose_x = x + facing * self.sx(8 * scale)
        draw.arc(
            (
                nose_x - self.sx(12 * scale),
                head_y + self.sy(10 * scale),
                nose_x + self.sx(16 * scale),
                head_y + self.sy(42 * scale),
            ),
            20 if facing > 0 else 160,
            205 if facing > 0 else 345,
            fill=shadow,
            width=max(1, self.sc(4 * scale)),
        )
        self._mouth(draw, x, head_y + self.sy(62 * scale), mouth, scale)

        shoulder_y = y + self.sy(105 * scale)
        arm_length = self.sx(150 * scale)
        gesture_base = _ease_out_cubic(min(1.0, local_progress * 3.0))
        emphasis = 0.5 + 0.5 * math.sin(t * 2.8 + phase)
        gesture = min(1.0, gesture_base * (0.84 + 0.16 * emphasis))
        left_shoulder = (x - self.sx(87 * scale), shoulder_y)
        right_shoulder = (x + self.sx(87 * scale), shoulder_y)
        if action in {"point", "explain"}:
            hand_x = x + facing * _lerp(self.sx(95 * scale), self.sx(240 * scale), gesture)
            hand_y = shoulder_y - _lerp(self.sy(5 * scale), self.sy(135 * scale), gesture)
            start = right_shoulder if facing > 0 else left_shoulder
            draw.line((start[0], start[1], hand_x, hand_y), fill=outline, width=self.sc(13 * scale))
            draw.ellipse((hand_x - self.sc(16), hand_y - self.sc(16), hand_x + self.sc(16), hand_y + self.sc(16)), fill=color, outline=outline, width=self.sc(5 * scale))
            other = left_shoulder if facing > 0 else right_shoulder
            draw.line((other[0], other[1], other[0] - facing * arm_length * 0.5, other[1] + self.sy(110 * scale)), fill=outline, width=self.sc(12 * scale))
        elif action == "think":
            hand_x = x + facing * self.sx(80 * scale)
            hand_y = head_y + self.sy(75 * scale)
            start = right_shoulder if facing > 0 else left_shoulder
            draw.line((start[0], start[1], hand_x, hand_y), fill=outline, width=self.sc(13 * scale))
            draw.ellipse((hand_x - self.sc(16), hand_y - self.sc(16), hand_x + self.sc(16), hand_y + self.sc(16)), fill=color, outline=outline, width=self.sc(5 * scale))
        elif action == "write":
            hand_x = x + facing * self.sx(115 * scale)
            hand_y = self.sy(1345)
            start = right_shoulder if facing > 0 else left_shoulder
            draw.line((start[0], start[1], hand_x, hand_y), fill=outline, width=self.sc(13 * scale))
            draw.line((hand_x, hand_y, hand_x + facing * self.sx(55 * scale), hand_y - self.sy(25 * scale)), fill=self.palette.accent, width=self.sc(7 * scale))
        elif action == "react":
            lift = self.sy((24 + 22 * emphasis) * scale)
            hand_x = x + facing * self.sx(118 * scale)
            hand_y = shoulder_y - lift
            start = right_shoulder if facing > 0 else left_shoulder
            draw.line((start[0], start[1], hand_x, hand_y), fill=outline, width=self.sc(12 * scale))
            draw.ellipse(
                (
                    hand_x - self.sc(13 * scale),
                    hand_y - self.sc(13 * scale),
                    hand_x + self.sc(13 * scale),
                    hand_y + self.sc(13 * scale),
                ),
                fill=color,
                outline=outline,
                width=max(1, self.sc(4 * scale)),
            )
            other = left_shoulder if facing > 0 else right_shoulder
            draw.line(
                (
                    other[0],
                    other[1],
                    other[0] - facing * self.sx(38 * scale),
                    other[1] + self.sy(115 * scale),
                ),
                fill=outline,
                width=self.sc(12 * scale),
            )
        else:
            draw.line((left_shoulder[0], left_shoulder[1], left_shoulder[0] - self.sx(40 * scale), left_shoulder[1] + self.sy(120 * scale)), fill=outline, width=self.sc(12 * scale))
            draw.line((right_shoulder[0], right_shoulder[1], right_shoulder[0] + self.sx(40 * scale), right_shoulder[1] + self.sy(120 * scale)), fill=outline, width=self.sc(12 * scale))

    def _panel(self, draw: ImageDraw.ImageDraw, progress: float) -> tuple[int, int, int, int]:
        ease = _ease_out_cubic(min(1.0, progress * 4.0))
        target = (self.sx(120), self.sy(330), self.sx(960), self.sy(1180))
        cx = (target[0] + target[2]) / 2
        cy = (target[1] + target[3]) / 2
        scale = 0.86 + 0.14 * ease
        box = (
            int(cx - (target[2] - target[0]) * scale / 2),
            int(cy - (target[3] - target[1]) * scale / 2),
            int(cx + (target[2] - target[0]) * scale / 2),
            int(cy + (target[3] - target[1]) * scale / 2),
        )
        draw.rounded_rectangle(box, radius=self.sc(55), fill=self.palette.card, outline=self.palette.ink, width=self.sc(9))
        return box

    def _overlay(self, draw: ImageDraw.ImageDraw, scene: CartoonScene, progress: float, t: float) -> None:
        if scene.overlay == "none":
            return
        box = self._panel(draw, progress)
        left, top, right, bottom = box
        cx = (left + right) // 2
        title_font = _font(self.sc(56))
        huge_font = _font(self.sc(102))
        small_font = _font(self.sc(34))
        ink = self.palette.card_ink
        accent = self.palette.host if scene.speaker == "host" else self.palette.guest
        overlay_text = scene.overlay_text or scene.accent_text or "IDEA"

        if scene.overlay in {"stat", "money"}:
            icon_y = top + self.sy(120)
            if scene.overlay == "money":
                for offset in (-70, 0, 70):
                    draw.ellipse((cx + self.sx(offset) - self.sx(42), icon_y - self.sy(35), cx + self.sx(offset) + self.sx(42), icon_y + self.sy(35)), fill=self.palette.accent, outline=ink, width=self.sc(5))
            _text_centered(draw, (cx, top + self.sy(320)), overlay_text, huge_font, ink)
            _text_centered(draw, (cx, top + self.sy(455)), "THE NUMBER THAT MATTERS", small_font, "#5D5F6E")
            draw.rounded_rectangle((left + self.sx(110), top + self.sy(590), right - self.sx(110), top + self.sy(630)), radius=self.sc(20), fill="#E3DED6")
            fill_width = int((right - left - self.sx(220)) * (0.55 + 0.35 * (0.5 + 0.5 * math.sin(t * 0.8))))
            draw.rounded_rectangle((left + self.sx(110), top + self.sy(590), left + self.sx(110) + fill_width, top + self.sy(630)), radius=self.sc(20), fill=accent)
            return

        if scene.overlay == "browser_tabs":
            browser = (left + self.sx(70), top + self.sy(120), right - self.sx(70), bottom - self.sy(120))
            draw.rounded_rectangle(browser, radius=self.sc(32), fill="#FFFFFF", outline=ink, width=self.sc(6))
            draw.rectangle((browser[0], browser[1], browser[2], browser[1] + self.sy(120)), fill="#E8E9EF")
            for index in range(5):
                tab_left = browser[0] + self.sx(25 + index * 130)
                fill = "#FFFFFF" if index == 3 else "#D1D4DE"
                draw.rounded_rectangle((tab_left, browser[1] + self.sy(28), tab_left + self.sx(110), browser[1] + self.sy(105)), radius=self.sc(18), fill=fill, outline="#A8ABB8", width=self.sc(3))
                draw.ellipse((tab_left + self.sx(12), browser[1] + self.sy(52), tab_left + self.sx(28), browser[1] + self.sy(68)), fill=accent)
            draw.rounded_rectangle((browser[0] + self.sx(70), browser[1] + self.sy(180), browser[2] - self.sx(70), browser[1] + self.sy(230)), radius=self.sc(18), fill="#ECEEF3")
            for row in range(5):
                width = self.sx(520 - row * 45)
                draw.rounded_rectangle((browser[0] + self.sx(95), browser[1] + self.sy(320 + row * 75), browser[0] + self.sx(95) + width, browser[1] + self.sy(348 + row * 75)), radius=self.sc(12), fill="#B9BDCA")
            _text_centered(draw, (cx, bottom - self.sy(85)), overlay_text, small_font, ink)
            return

        if scene.overlay == "email":
            draw.rounded_rectangle((left + self.sx(75), top + self.sy(115), right - self.sx(75), bottom - self.sy(115)), radius=self.sc(30), fill="#FFFFFF", outline=ink, width=self.sc(6))
            draw.rectangle((left + self.sx(75), top + self.sy(115), right - self.sx(75), top + self.sy(245)), fill="#E9EBF0")
            draw.text((left + self.sx(115), top + self.sy(155)), "COMPOSE", font=small_font, fill=ink)
            draw.rounded_rectangle((right - self.sx(300), top + self.sy(150), right - self.sx(120), top + self.sy(215)), radius=self.sc(18), fill=self.palette.accent)
            draw.text((right - self.sx(255), top + self.sy(165)), "DRAFT", font=_font(self.sc(26)), fill=ink)
            for row, width in enumerate((510, 430, 560, 350)):
                draw.rounded_rectangle((left + self.sx(125), top + self.sy(340 + row * 95), left + self.sx(125 + width), top + self.sy(375 + row * 95)), radius=self.sc(14), fill="#C9CCD6")
            draw.line((left + self.sx(125), top + self.sy(725), left + self.sx(340), top + self.sy(725)), fill=accent, width=self.sc(7))
            _text_centered(draw, (cx, bottom - self.sy(80)), overlay_text, small_font, ink)
            return

        if scene.overlay == "checklist":
            draw.text((left + self.sx(110), top + self.sy(120)), overlay_text or "NEXT STEP", font=title_font, fill=ink)
            for index, label in enumerate(("OPEN TASK", "WRITE NEXT STEP", "RETURN LATER")):
                y = top + self.sy(310 + index * 190)
                draw.rounded_rectangle((left + self.sx(115), y, left + self.sx(185), y + self.sy(70)), radius=self.sc(14), outline=ink, width=self.sc(6), fill="#FFFFFF")
                if index < 2:
                    draw.line((left + self.sx(130), y + self.sy(38), left + self.sx(150), y + self.sy(58), left + self.sx(178), y + self.sy(18)), fill=self.palette.green, width=self.sc(9), joint="curve")
                draw.text((left + self.sx(225), y + self.sy(10)), label, font=small_font, fill=ink)
            return

        if scene.overlay == "brain":
            center_y = top + self.sy(420)
            nodes = [(-180, -90), (-80, -170), (60, -145), (175, -60), (-150, 80), (-20, 40), (135, 105)]
            for i, (dx, dy) in enumerate(nodes):
                for j in range(i + 1, len(nodes)):
                    dx2, dy2 = nodes[j]
                    if abs(dx - dx2) + abs(dy - dy2) < 260:
                        draw.line((cx + self.sx(dx), center_y + self.sy(dy), cx + self.sx(dx2), center_y + self.sy(dy2)), fill="#9599A9", width=self.sc(5))
            for index, (dx, dy) in enumerate(nodes):
                radius = self.sx(38 if index != 5 else 55)
                fill = accent if index == 5 else self.palette.cyan
                draw.ellipse((cx + self.sx(dx) - radius, center_y + self.sy(dy) - radius, cx + self.sx(dx) + radius, center_y + self.sy(dy) + radius), fill=fill, outline=ink, width=self.sc(5))
            _text_centered(draw, (cx, bottom - self.sy(170)), overlay_text, title_font, ink)
            return

        if scene.overlay == "phone":
            phone = (cx - self.sx(185), top + self.sy(90), cx + self.sx(185), bottom - self.sy(90))
            draw.rounded_rectangle(phone, radius=self.sc(55), fill="#252936", outline=ink, width=self.sc(9))
            screen = (phone[0] + self.sx(22), phone[1] + self.sy(65), phone[2] - self.sx(22), phone[3] - self.sy(65))
            draw.rounded_rectangle(screen, radius=self.sc(35), fill="#F8F8FA")
            for index in range(4):
                y = screen[1] + self.sy(90 + index * 145)
                draw.rounded_rectangle((screen[0] + self.sx(35), y, screen[2] - self.sx(35), y + self.sy(105)), radius=self.sc(28), fill="#E8EAF0")
                draw.ellipse((screen[0] + self.sx(55), y + self.sy(26), screen[0] + self.sx(95), y + self.sy(66)), fill=accent)
            badge_r = self.sx(58)
            draw.ellipse((phone[2] - badge_r, phone[1] - badge_r / 2, phone[2] + badge_r, phone[1] + badge_r * 1.5), fill=self.palette.red, outline=ink, width=self.sc(5))
            _text_centered(draw, (phone[2], phone[1] - self.sy(15)), "4", _font(self.sc(48)), "white")
            return

        if scene.overlay == "clock":
            radius = self.sx(245)
            cy = top + self.sy(420)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="#FFFFFF", outline=ink, width=self.sc(9))
            for index in range(12):
                angle = math.radians(index * 30 - 90)
                x1 = cx + math.cos(angle) * radius * 0.78
                y1 = cy + math.sin(angle) * radius * 0.78
                x2 = cx + math.cos(angle) * radius * 0.9
                y2 = cy + math.sin(angle) * radius * 0.9
                draw.line((x1, y1, x2, y2), fill=ink, width=self.sc(5))
            phase = (t * 0.22) % 1.0
            minute_angle = math.radians(phase * 360 - 90)
            draw.line((cx, cy, cx + math.cos(minute_angle) * radius * 0.68, cy + math.sin(minute_angle) * radius * 0.68), fill=accent, width=self.sc(10))
            draw.line((cx, cy, cx - self.sx(80), cy - self.sy(95)), fill=ink, width=self.sc(13))
            _text_centered(draw, (cx, bottom - self.sy(120)), overlay_text, title_font, ink)
            return

        if scene.overlay in {"chart", "comparison"}:
            if scene.overlay == "comparison":
                card_w = self.sx(290)
                for index, (label, fill) in enumerate((("A", self.palette.host), ("B", self.palette.guest))):
                    x = cx - self.sx(330) + index * self.sx(370)
                    draw.rounded_rectangle((x, top + self.sy(210), x + card_w, bottom - self.sy(220)), radius=self.sc(35), fill=fill, outline=ink, width=self.sc(7))
                    _text_centered(draw, (x + card_w / 2, top + self.sy(360)), label, huge_font, "white")
                _text_centered(draw, (cx, bottom - self.sy(140)), overlay_text, title_font, ink)
            else:
                baseline = bottom - self.sy(210)
                heights = (210, 350, 290, 470, 580)
                for index, height in enumerate(heights):
                    x = left + self.sx(120 + index * 125)
                    draw.rounded_rectangle((x, baseline - self.sy(height), x + self.sx(75), baseline), radius=self.sc(16), fill=accent if index == 4 else self.palette.cyan, outline=ink, width=self.sc(4))
                draw.line((left + self.sx(90), baseline, right - self.sx(80), baseline), fill=ink, width=self.sc(6))
                _text_centered(draw, (cx, top + self.sy(80)), overlay_text, title_font, ink)
            return

        # thought (and unknown sanitized fallback)
        bubble = (left + self.sx(90), top + self.sy(150), right - self.sx(90), bottom - self.sy(260))
        draw.rounded_rectangle(bubble, radius=self.sc(90), fill="#FFFFFF", outline=ink, width=self.sc(7))
        draw.ellipse((left + self.sx(210), bottom - self.sy(260), left + self.sx(290), bottom - self.sy(180)), fill="#FFFFFF", outline=ink, width=self.sc(5))
        draw.ellipse((left + self.sx(160), bottom - self.sy(165), left + self.sx(210), bottom - self.sy(115)), fill="#FFFFFF", outline=ink, width=self.sc(4))
        lines = _wrap_lines(overlay_text, width=18, max_lines=3)
        y = top + self.sy(360)
        for line in lines:
            _text_centered(draw, (cx, y), line, title_font, ink)
            y += self.sy(90)

    def _compact_overlay(
        self,
        image: Image.Image,
        scene: CartoonScene,
        progress: float,
        t: float,
    ) -> None:
        """Reuse the full explanatory graphic as a picture-in-picture concept card."""
        if scene.overlay == "none":
            return
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self._overlay(ImageDraw.Draw(layer), scene, progress, t)
        source = (
            self.sx(85),
            self.sy(285),
            self.sx(995),
            self.sy(1215),
        )
        card = layer.crop(source)
        max_width = self.sx(820)
        max_height = self.sy(600)
        factor = min(max_width / card.width, max_height / card.height)
        size = (
            max(1, int(round(card.width * factor))),
            max(1, int(round(card.height * factor))),
        )
        card = card.resize(size, Image.Resampling.LANCZOS)
        left = (self.width - card.width) // 2
        top = self.sy(62)
        image.paste(card, (left, top), card)

    def render(self, t: float) -> Image.Image:
        scene = _scene_for_time(self.scenes, t)
        local = max(0.0, min(scene.duration, t - scene.start))
        progress = local / scene.duration
        image = Image.new("RGB", (self.width, self.height), self.palette.background)
        draw = ImageDraw.Draw(image)
        self._background(draw, t)
        self._table(draw)

        speaking_host = scene.speaker == "host"
        mouth = _mouth_at(self.mouth_cues, t, True)
        closed = "X"
        # Graphic layouts reserve the center for the explanatory card and keep a small
        # presenter at the desk edge so the format still feels like the same show.
        if scene.layout == "graphic":
            self._overlay(draw, scene, progress, t)
            presenter_is_host = scene.speaker == "host"
            x = self.sx(165 if presenter_is_host else 915)
            self._character(
                draw,
                x=x,
                base_y=self.sy(1280),
                color=self.palette.host if presenter_is_host else self.palette.guest,
                shadow=self.palette.host_shadow if presenter_is_host else self.palette.guest_shadow,
                emotion=scene.emotion,
                action="react",
                mouth=mouth,
                facing=1 if presenter_is_host else -1,
                t=t,
                local_progress=progress,
                scale=0.62,
            )
            return image

        if scene.layout == "two_shot":
            has_overlay = scene.overlay != "none"
            pair_base_y = self.sy(1040 if has_overlay else 980)
            pair_scale = 0.84 if has_overlay else 0.95
            self._microphone(
                draw,
                self.sx(390),
                self.sy(1215 if has_overlay else 1180),
                flip=False,
            )
            self._microphone(
                draw,
                self.sx(690),
                self.sy(1215 if has_overlay else 1180),
                flip=True,
            )
            self._character(
                draw,
                x=self.sx(285),
                base_y=pair_base_y,
                color=self.palette.host,
                shadow=self.palette.host_shadow,
                emotion=scene.emotion if speaking_host else "neutral",
                action=scene.action if speaking_host else "react",
                mouth=mouth if speaking_host else closed,
                facing=1,
                t=t,
                local_progress=progress,
                scale=pair_scale,
                dimmed=not speaking_host,
            )
            self._character(
                draw,
                x=self.sx(795),
                base_y=pair_base_y,
                color=self.palette.guest,
                shadow=self.palette.guest_shadow,
                emotion=scene.emotion if not speaking_host else "neutral",
                action=scene.action if not speaking_host else "react",
                mouth=mouth if not speaking_host else closed,
                facing=-1,
                t=t,
                local_progress=progress,
                scale=pair_scale,
                dimmed=speaking_host,
            )
        else:
            host_layout = scene.layout == "host"
            color = self.palette.host if host_layout else self.palette.guest
            shadow = self.palette.host_shadow if host_layout else self.palette.guest_shadow
            facing = 1 if host_layout else -1
            has_overlay = scene.overlay != "none"
            presenter_base_y = self.sy(1040 if has_overlay else 860)
            presenter_scale = 0.96 if has_overlay else 1.26
            self._microphone(
                draw,
                self.sx(660 if host_layout else 420),
                self.sy(1210 if has_overlay else 1085),
                flip=not host_layout,
            )
            self._character(
                draw,
                x=self.sx(470 if host_layout else 610),
                base_y=presenter_base_y,
                color=color,
                shadow=shadow,
                emotion=scene.emotion,
                action=scene.action,
                mouth=mouth,
                facing=facing,
                t=t,
                local_progress=progress,
                scale=presenter_scale,
            )

        # Presenter and two-shot layouts keep the recurring cast prominent while the
        # actual semantic visual (browser, email, checklist, brain, etc.) appears above.
        if scene.overlay != "none":
            self._compact_overlay(image, scene, progress, t)
        return image


def _write_storyboard(
    renderer: DoodleRenderer,
    scenes: list[CartoonScene],
    destination: Path,
) -> None:
    thumbs: list[Image.Image] = []
    for scene in scenes[:12]:
        t = min(scene.end - 0.05, scene.start + min(0.8, scene.duration * 0.35))
        frame = renderer.render(max(scene.start, t))
        thumb = frame.copy()
        thumb.thumbnail((270, 480), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (290, 530), "#0B0C12")
        canvas.paste(thumb, ((290 - thumb.width) // 2, 10))
        label_draw = ImageDraw.Draw(canvas)
        label_draw.text((12, 490), f"Scene {scene.scene} · {scene.overlay}", font=_font(20), fill="white")
        thumbs.append(canvas)
    if not thumbs:
        return
    columns = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 300, rows * 540), "#11131D")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 300 + 5
        y = (index // columns) * 540 + 5
        sheet.paste(thumb, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=88, optimize=True)


def _resolution(aspect: VideoAspect | str) -> tuple[int, int]:
    try:
        resolved = aspect if isinstance(aspect, VideoAspect) else VideoAspect(aspect)
    except ValueError:
        resolved = VideoAspect.portrait
    return resolved.to_resolution()


def _render_raw_frames(
    *,
    renderer: DoodleRenderer,
    duration: float,
    fps: int,
    output_path: Path,
) -> None:
    ffmpeg = utils.get_ffmpeg_binary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{renderer.width}x{renderer.height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    logger.info(
        "rendering AI cartoon material: "
        f"duration={duration:.2f}s, fps={fps}, resolution={renderer.width}x{renderer.height}"
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise CartoonRenderError(f"failed to start ffmpeg cartoon renderer: {exc}") from exc

    frame_count = max(1, int(math.ceil(duration * fps)))
    render_error: Exception | None = None
    try:
        if process.stdin is None:
            raise CartoonRenderError("ffmpeg cartoon renderer has no stdin pipe")
        for frame_index in range(frame_count):
            t = min(duration, frame_index / fps)
            frame = renderer.render(t)
            process.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError, CartoonRenderError) as exc:
        render_error = exc
    finally:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
    stderr_bytes = b""
    if process.stderr is not None:
        stderr_bytes = process.stderr.read()
    return_code = process.wait()
    if render_error is not None:
        raise CartoonRenderError(f"cartoon frame stream failed: {render_error}") from render_error
    if return_code != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        detail = stderr_bytes.decode("utf-8", errors="replace")[-1000:]
        raise CartoonRenderError(f"ffmpeg cartoon render failed: {detail or return_code}")


def render_cartoon_material(
    *,
    task_id: str,
    params: VideoParams,
    video_script: str,
    audio_file: str,
    audio_duration: float,
    subtitle_path: str = "",
) -> str:
    """Render one full-length, silent cartoon video material for the task."""
    task_dir = Path(utils.task_dir(task_id))
    fps = int(getattr(params, "cartoon_fps", DEFAULT_FPS) or DEFAULT_FPS)
    fps = max(12, min(30, fps))
    width, height = _resolution(params.video_aspect)
    # Full 1080x1920 output is intentional: this source is vector-like and benefits
    # from rendering at delivery resolution rather than upscaling a smaller raster.
    scenes, director = build_cartoon_plan(
        subject=str(params.video_subject or ""),
        script=video_script,
        audio_duration=float(audio_duration),
        subtitle_path=subtitle_path,
        ai_director=bool(getattr(params, "cartoon_ai_director", True)),
        app_config=config.snapshot_config_with_pending(config.app),
    )
    if not scenes:
        raise CartoonRenderError("cartoon director produced no scenes")

    lip_sync_mode = str(getattr(params, "cartoon_lip_sync", "auto") or "auto").lower()
    if lip_sync_mode not in {"auto", "rhubarb", "heuristic"}:
        lip_sync_mode = "auto"
    mouth_cues, lip_sync_backend = generate_mouth_cues(audio_file, mode=lip_sync_mode)
    renderer = DoodleRenderer(width, height, scenes, mouth_cues)

    plan_payload = {
        "schema": CARTOON_PLAN_SCHEMA,
        "schema_version": CARTOON_PLAN_VERSION,
        "task_id": task_id,
        "style": "original_doodle_podcast",
        "director": director,
        "lip_sync": lip_sync_backend,
        "fps": fps,
        "resolution": [width, height],
        "duration": float(audio_duration),
        "scenes": [scene.to_dict() for scene in scenes],
    }
    plan_path = task_dir / "cartoon_plan.json"
    plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    storyboard_path = task_dir / "cartoon_storyboard.jpg"
    _write_storyboard(renderer, scenes, storyboard_path)

    output_path = task_dir / "cartoon-material.mp4"
    _render_raw_frames(
        renderer=renderer,
        duration=max(float(audio_duration), scenes[-1].end),
        fps=fps,
        output_path=output_path,
    )
    task_artifacts.patch_script_data(
        task_id,
        cartoon_animation={
            "available": True,
            "style": "original_doodle_podcast",
            "director": director,
            "lip_sync": lip_sync_backend,
            "fps": fps,
            "scene_count": len(scenes),
            "plan_file": plan_path.name,
            "storyboard_file": storyboard_path.name,
            "material_file": output_path.name,
        },
    )
    logger.success(
        "AI cartoon material complete: "
        f"task_id={task_id}, scenes={len(scenes)}, director={director}, lip_sync={lip_sync_backend}"
    )
    return str(output_path)
