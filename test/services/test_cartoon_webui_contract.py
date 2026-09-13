from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def test_ai_cartoon_is_accepted_by_generation_validation():
    text = WEBUI_MAIN.read_text(encoding="utf-8")
    validation_start = text.index("if params.video_source not in [")
    validation_end = text.index("]:", validation_start)
    validation = text[validation_start:validation_end]
    assert '"ai_cartoon"' in validation


def test_ai_cartoon_concat_widget_is_visibly_locked_to_sequential():
    text = WEBUI_MAIN.read_text(encoding="utf-8")
    assert 'key="video_concat_mode_cartoon_locked"' in text
    assert "options=[VideoConcatMode.sequential.value]" in text
    assert "AI Cartoon uses one authored narration timeline in sequential order." in text
