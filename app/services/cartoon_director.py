"""Small validated storytelling vocabulary, independent of drawing and LLM I/O.

Templates are executable contracts, not free-form frame descriptions. The local
fallback recognizes a few EN/ES event relations and uses prior state for pronouns;
unsupported meanings retain the existing explanatory renderer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StoryTemplate:
    focus_object: str
    visual_action: str
    before: tuple[str, ...]
    after: str
    overlay: str
    pose: str


TEMPLATES = {
    "explain_generic": StoryTemplate("none", "explain", ("neutral",), "neutral", "none", "explain"),
    "stop_work": StoryTemplate("task", "set_aside", ("working",), "unfinished", "email", "react"),
    "mental_persistence": StoryTemplate("thought", "keep_active", ("unfinished", "mentally_active"), "mentally_active", "brain", "think"),
    "browser_open_tab": StoryTemplate("browser", "isolate_pending", ("mentally_active", "unfinished"), "open_tab", "browser_tabs", "point"),
    "unfinished_email": StoryTemplate("email", "interrupt_draft", ("open_tab", "unfinished", "mentally_active"), "draft", "email", "write"),
    "write_next_step": StoryTemplate("note", "record", ("draft", "unfinished", "mentally_active"), "recorded", "checklist", "write"),
    "task_parked": StoryTemplate("return_slot", "store_for_return", ("recorded",), "parked", "checklist", "point"),
    "mental_release": StoryTemplate("thought", "release_attention", ("parked", "recorded"), "released", "brain", "react"),
}


def fallback_template(text: str, previous_state: str = "neutral") -> str:
    """Conservative relations, with negation and context; never whole-fixture matching."""
    text = text.casefold()
    if re.search(r"\b(?:do not|don't|never|no debes|no hay que)\b", text):
        return "explain_generic"
    relations = (
        ("write_next_step", r"\b(?:write|record|jot|anota|escribe|registra)\w*\b.{0,70}\b(?:step|idea|appointment|note|paso|idea|cita)\b"),
        ("mental_release", r"\b(?:attention|mind|focus|atención|mente)\b.{0,55}\b(?:move on|relax|free|rest|descans|liber)\w*"),
        ("task_parked", r"\b(?:place|point|location|lugar|sitio)\b.{0,30}\b(?:return|volver|regresar)\b"),
        ("unfinished_email", r"\b(?:unfinished|half.written|draft|incomplete|borrador|incomplet\w*)\b.{0,25}\b(?:email|message|correo|mensaje)\b"),
        ("unfinished_email", r"\b(?:email|message|correo|mensaje)\b.{0,35}\b(?:unfinished|draft|incomplete|borrador|incomplet\w*)\b"),
        ("browser_open_tab", r"\b(?:browser|tab|pestaña|navegador)\b.{0,55}\b(?:open|closed|pending|waiting|abiert\w*|cerrar)\b"),
        ("mental_persistence", r"\b(?:task|work|idea|tarea|trabajo)\b.{0,65}\b(?:head|mind|active|mente|cabeza|activ\w*)\b"),
        ("stop_work", r"\b(?:stop|stopped|pause|leave|deja|parar|dejar)\b.{0,25}\b(?:work|working|desk|trabaj\w*)\b"),
    )
    for template, pattern in relations:
        if re.search(pattern, text):
            return template
    # A pronoun-only storage instruction needs an established recorded object.
    if previous_state == "recorded" and re.search(
        r"\b(?:save|store|keep|guarda)\w*\b.{0,35}\b(?:later|return|después|luego)\b", text
    ):
        return "task_parked"
    return "explain_generic"


def story_fields(template: str, previous_state: str = "neutral") -> dict[str, str]:
    spec = TEMPLATES[template]
    return dict(
        scene_template=template, focus_object=spec.focus_object,
        visual_action=spec.visual_action,
        state_before=previous_state if previous_state in spec.before else spec.before[0],
        state_after=spec.after,
        continuity_object="" if template == "explain_generic" else "pending_item_1",
    )


def validated_fields(raw: dict, fallback, previous_state: str = "neutral") -> dict[str, str]:
    name = raw.get("scene_template", fallback.scene_template)
    if not isinstance(name, str) or name not in TEMPLATES:
        name = fallback.scene_template
    fields = story_fields(name, previous_state)
    # Coupled decisions must agree with a single template. Repair the whole
    # transition to the local semantic choice rather than combining contradictions.
    spec = TEMPLATES[name]
    for key, value in fields.items():
        if key not in raw:
            continue
        if key == "state_before" and raw[key] in spec.before:
            fields[key] = raw[key]
        elif raw[key] != value:
            return story_fields(fallback.scene_template, previous_state)
    return fields
