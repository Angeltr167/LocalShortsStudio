from app.services import cartoon_engine as engine


def scene(layout, visual_actor, narration_actor="host"):
    return engine.CartoonScene(
        scene=1, start=0, end=2, narration="Explain this.", layout=layout,
        speaker="guest" if visual_actor == "guest" else "host", emotion="neutral",
        action="react", overlay="none", narration_actor=narration_actor,
        visual_actor=visual_actor, actor_role="reaction" if visual_actor == "guest" else "narrator",
        supporting_character="host" if visual_actor == "guest" else "guest",
    )


def test_single_voice_fallback_always_assigns_host_narrator():
    cue = engine.NarrationCue(0, 2, "The guest reacts.")
    fallback = engine._fallback_scene(cue, 0)
    assert fallback.narration_actor == "host"
    assert fallback.supporting_character in {"host", "guest"}


def test_ai_cannot_transfer_narration_ownership_to_guest():
    cue = engine.NarrationCue(0, 2, "The guest reacts.")
    fallback = engine._fallback_scene(cue, 0)
    directed = engine._sanitize_scene_choice(
        {"speaker": "guest", "visual_actor": "guest", "actor_role": "narrator"}, fallback
    )
    assert directed.narration_actor == "host"
    assert directed.visual_actor == "guest"
    assert directed.actor_role == "reaction"


def test_all_layouts_have_explicit_actor_roles():
    for layout, actor in (("host", "host"), ("guest", "guest"),
                          ("two_shot", "host"), ("graphic", "host")):
        item = scene(layout, actor)
        assert item.narration_actor == "host"
        assert item.visual_actor == actor
        assert item.supporting_character != item.visual_actor


def test_mouth_owner_is_independent_from_visual_actor():
    renderer = engine.DoodleRenderer(180, 320, [scene("guest", "guest")],
                                     [engine.MouthCue(0, 1, "B")])
    item = renderer.scenes[0]
    assert renderer.mouth_for_character(item, "host", 0.5) == "B"
    assert renderer.mouth_for_character(item, "guest", 0.5) == "X"


def test_storytelling_adapter_uses_narrator_role():
    item = scene("two_shot", "guest")
    renderer = engine.DoodleRenderer(180, 320, [item], [engine.MouthCue(0, 1, "B")])
    assert renderer.mouth_for_character(item, "host", 0.5) == "B"
    assert renderer.mouth_for_character(item, "guest", 0.5) == "X"
