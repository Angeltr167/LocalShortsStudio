"""Run the production task pipeline with fresh or preserved local narration.

Reference mode supplies the existing supported full-voice-preview input. It does
not mock generation, replace the planner, or bypass the final compositor.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schema import VideoParams  # noqa: E402
from app.services import cartoon_engine, task, voice  # noqa: E402
from app.utils import utils  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--mode", choices=("reference", "fallback", "e2e"), default="reference")
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_dir.resolve()
    record = json.loads((reference / "script.json").read_text(encoding="utf-8"))
    params = VideoParams(**record["params"])
    params.video_source = "ai_cartoon"
    params.video_script = record["script"]
    params.video_count = 1
    params.cartoon_ai_director = args.mode != "fallback" and args.phase > 1
    params.cartoon_lip_sync = "heuristic"
    params.voice_name = "chatterbox:default-Female"
    params.voice_volume = 1.0
    params.bgm_type = ""
    params.custom_audio_file = None
    task_id = f"cartoon-v2-phase{args.phase}-{args.mode}-{uuid4().hex[:8]}"
    destination = Path(utils.task_dir(task_id))
    preview = None
    if args.mode != "e2e":
        audio = destination / "audio.mp3"
        shutil.copy2(reference / "audio.mp3", audio)
        cues = cartoon_engine.parse_srt_cues(reference / "subtitle.srt")
        maker = SimpleNamespace(
            cues=[], subs=[cue.text for cue in cues],
            offset=[(round(cue.start * 10_000_000), round(cue.end * 10_000_000)) for cue in cues],
        )
        preview = dict(
            script=params.video_script.strip(), voice_name=params.voice_name,
            voice_rate=float(params.voice_rate), voice_volume=1.0,
            audio_file=str(audio), duration=voice.get_audio_duration(str(audio)),
            sub_maker=maker,
        )
    result = task.start(task_id, params, voice_preview=preview)
    required = ["cartoon_plan.json", "cartoon_storyboard.jpg", "cartoon-material.mp4", "final-1.mp4"]
    missing = [name for name in required if not (destination / name).is_file()]
    report = dict(phase=args.phase, mode=args.mode, output=str(destination), missing=missing,
                  result=result, timing_note="Reference SRT boundaries are not phoneme measurements.")
    (destination / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, default=str))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
