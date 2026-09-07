from pathlib import Path

path = Path("app/services/strict_scene.py")
text = path.read_text(encoding="utf-8")
old = """    return plan\n\n\ndef _mark_semantic_bridges(\n"""
new = """    _mark_semantic_bridges(plan, terms)\n    return plan\n\n\ndef _mark_semantic_bridges(\n"""
assert old in text, "build_scene_plan bridge compatibility marker not found"
path.write_text(text.replace(old, new, 1), encoding="utf-8")
