from unittest.mock import patch

from app.services import cartoon_engine


def test_mouth_cue_smoothing_sorts_merges_and_drops_subframe_noise():
    cues = cartoon_engine._smooth_mouth_cues(
        [
            cartoon_engine.MouthCue(0.22, 0.30, "A"),
            cartoon_engine.MouthCue(0.00, 0.10, "A"),
            cartoon_engine.MouthCue(0.10, 0.115, "A"),
            cartoon_engine.MouthCue(0.31, 0.40, "C"),
        ]
    )
    assert cues == [
        cartoon_engine.MouthCue(0.00, 0.10, "A"),
        cartoon_engine.MouthCue(0.22, 0.30, "A"),
        cartoon_engine.MouthCue(0.31, 0.40, "C"),
    ]


def test_rhubarb_parser_returns_time_ordered_smoothed_cues():
    cues = cartoon_engine._parse_rhubarb_payload(
        {
            "mouthCues": [
                {"start": 0.20, "end": 0.30, "value": "A"},
                {"start": 0.00, "end": 0.10, "value": "A"},
                {"start": 0.10, "end": 0.20, "value": "A"},
            ]
        }
    )
    assert cues == [cartoon_engine.MouthCue(0.00, 0.30, "A")]


def test_auto_lipsync_keeps_deterministic_heuristic_fallback_when_audio_analysis_fails():
    with (
        patch.object(cartoon_engine, "_resolve_rhubarb_binary", return_value=""),
        patch.object(cartoon_engine, "_heuristic_audio_mouth_cues", return_value=[]),
    ):
        cues, backend = cartoon_engine.generate_mouth_cues("missing.mp3", mode="auto")
    assert cues == []
    assert backend == "heuristic"
    # No audio-derived cues still produce a stable closed/open choice for a
    # speaking actor, while a reaction actor stays closed.
    assert cartoon_engine._mouth_at([], 0.25, True) == cartoon_engine._mouth_at([], 0.25, True)
    assert cartoon_engine._mouth_at([], 0.25, False) == "X"
