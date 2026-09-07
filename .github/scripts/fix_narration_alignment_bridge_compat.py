from pathlib import Path

path = Path("app/services/strict_scene.py")
text = path.read_text(encoding="utf-8")
start = text.index("def build_scene_plan(")
marker = text.index("def _mark_semantic_bridges(", start)
segment = text[start:marker]
needle = "    return plan\n"
position = segment.rfind(needle)
assert position >= 0, "build_scene_plan return marker not found"
absolute = start + position
replacement = "    _mark_semantic_bridges(plan, terms)\n    return plan\n"
text = text[:absolute] + replacement + text[absolute + len(needle):]
path.write_text(text, encoding="utf-8")
