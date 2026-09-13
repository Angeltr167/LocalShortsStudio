from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services import cartoon_engine as cartoon
from app.models.schema import VideoParams
from app.services import task


FIXTURES = Path(__file__).parents[1] / "fixtures" / "cartoon_storytelling"


def assert_coverage(cues, duration):
    assert cues[0].start == 0
    assert cues[-1].end == duration
    assert all(cue.end > cue.start for cue in cues)
    assert all(a.end == b.start for a, b in zip(cues, cues[1:]))


def test_reference_script_preserves_instruction_and_payoff():
    script = (FIXTURES / "unfinished_task.txt").read_text(encoding="utf-8")
    cues = cartoon.build_narration_timeline(
        script, 21.34, str(FIXTURES / "unfinished_task.srt")
    )
    instruction = next(c for c in cues if "Before you stop working" in c.text)
    email = next(c for c in cues if "half-written email" in c.text)
    payoff = next(c for c in cues if "attention can finally move on" in c.text)
    assert "write down the exact next step" in instruction.text
    assert "Before" not in email.text
    assert email.end == instruction.start == pytest.approx(14.244)
    assert "clear place to return" not in payoff.text
    assert payoff.start == pytest.approx(19.451)
    assert 7 <= len(cues) <= 8
    assert_coverage(cues, 21.34)


@pytest.mark.parametrize("script", [
    "Stop.\n\nThink.\n\nReturn.",
    "Para.\n\nPiensa.\n\nVuelve.",
    "Uno。Dos！Tres？",
])
def test_short_sentences_and_paragraphs_survive(script):
    cues = cartoon.build_narration_timeline(script, 1.2)
    assert len(cues) == 3
    assert_coverage(cues, 1.2)


def test_one_unpunctuated_srt_cue_contains_multiple_script_sentences(tmp_path):
    path = tmp_path / "sub.srt"
    path.write_text("1\n00:00:00,000 --> 00:00:06,000\nStop now think later return tomorrow\n")
    cues = cartoon.build_narration_timeline("Stop now. Think later. Return tomorrow.", 6, str(path))
    assert [c.start for c in cues] == [0, 2, 4]
    assert cues[1].timing_source == "interpolated_srt"
    assert_coverage(cues, 6)


def test_word_cues_align_without_duration_merging(tmp_path):
    path = tmp_path / "sub.srt"
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\nStop\n\n"
                    "2\n00:00:01,000 --> 00:00:02,000\nnow\n\n"
                    "3\n00:00:02,000 --> 00:00:03,000\nRest\n")
    cues = cartoon.build_narration_timeline("Stop now. Rest.", 3, str(path))
    assert len(cues) == 2
    assert cues[1].start == 2
    assert cues[1].timing_source == "direct_srt"


@pytest.mark.parametrize("text", [
    "1\n00:00:01,000 --> 00:00:03,000\nStop now\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\nThen return\n",
    "1\n00:00:00,000 --> 00:00:01,000\nStop um now\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nreturn\n",
    "garbage",
])
def test_noisy_srt_retains_script_and_full_coverage(tmp_path, text):
    path = tmp_path / "sub.srt"
    path.write_text(text)
    script = "Stop now. Then return."
    first = cartoon.build_narration_timeline(script, 4.75, str(path))
    assert [c.text for c in first] == ["Stop now.", "Then return."]
    assert_coverage(first, 4.75)
    assert first == cartoon.build_narration_timeline(script, 4.75, str(path))


def test_bounded_clauses_do_not_split_intro_or_decimals():
    script = "Antes de salir, anota la idea. Tiene un lugar seguro, así que puedes descansar."
    cues = cartoon.build_narration_timeline(script, 8)
    assert len(cues) == 3
    assert cues[0].text == "Antes de salir, anota la idea."
    assert cues[-1].text.startswith("así que")
    assert len(cartoon.build_narration_timeline("Dr. Smith paid 3.14 dollars.", 9)) == 1


def test_empty_inputs_have_neutral_coverage():
    assert_coverage(cartoon.build_narration_timeline("", 0.2), 0.2)


def test_long_beat_is_not_split_for_duration():
    assert len(cartoon.build_narration_timeline("Keep holding the same position", 18)) == 1


@pytest.mark.parametrize("source, expected", [("ai_cartoon", 21.34), ("pexels", 22)])
def test_generated_audio_duration_is_exact_only_for_cartoon(tmp_path, source, expected):
    params = VideoParams(video_subject="test", video_source=source, voice_name="test")
    with (
        patch.object(task.utils, "task_dir", return_value=str(tmp_path)),
        patch.object(task.voice, "tts", return_value=MagicMock()),
        patch.object(task.voice, "get_audio_duration", return_value=21.34),
    ):
        _, duration, _ = task.generate_audio("test", params, "A narration.")
    assert duration == expected


def test_preview_duration_is_exact_for_cartoon(tmp_path):
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"fixture")
    params = VideoParams(video_subject="test", video_source="ai_cartoon", voice_volume=1.0)
    preview = dict(script="Hello", voice_name=params.voice_name, voice_rate=params.voice_rate,
                   voice_volume=1.0, audio_file=str(audio), duration=21.34, sub_maker=object())
    with patch.object(task.utils, "task_dir", return_value=str(tmp_path)):
        assert task._resolve_reusable_voice_preview("test", params, "Hello", preview)[1] == 21.34


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_fails_explicitly(duration):
    with pytest.raises(cartoon.CartoonRenderError):
        cartoon.build_narration_timeline("Hello", duration)


def test_missing_script_uses_srt(tmp_path):
    path = tmp_path / "subtitle.srt"
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\nOne\n\n"
                    "2\n00:00:02,000 --> 00:00:03,000\nTwo\n")
    cues = cartoon.build_narration_timeline("", 4, str(path))
    assert [cue.text for cue in cues] == ["One", "Two"]
    assert_coverage(cues, 4)
