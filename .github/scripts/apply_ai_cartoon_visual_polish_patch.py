from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"expected exactly one anchor in {path}, found {count}: {old[:140]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once('webui/Main.py', '            "openai_image",\n            "local",\n', '            "openai_image",\n            "ai_cartoon",\n            "local",\n')

replace_once('webui/Main.py', '            sync_script_order_concat_mode()\n            if params.video_source == "ai_cartoon":\n                st.session_state["video_concat_mode_select"] = VideoConcatMode.sequential.value\n            selected_concat_mode = stable_selectbox(\n                tr("Video Concat Mode"),\n                options=[value for _, value in video_concat_modes],\n                default_value=_saved_ui_choice(\n                    "video_concat_mode",\n                    [value for _, value in video_concat_modes],\n                    VideoConcatMode.random.value,\n                ),\n                key="video_concat_mode_select",\n                format_func=lambda value: dict(\n                    (v, label) for label, v in video_concat_modes\n                )[value],\n                disabled=bool(\n                    st.session_state.get("match_materials_to_script", False)\n                    or st.session_state.get("strict_scene_matching", False)\n                    or params.video_source == "ai_cartoon"\n                ),\n            )\n            params.video_concat_mode = VideoConcatMode(selected_concat_mode)\n', '            sync_script_order_concat_mode()\n            if params.video_source == "ai_cartoon":\n                # Use a dedicated widget key and a one-option list. Reusing the normal\n                # selectbox key can resurrect a persisted "random" value on Streamlit\n                # reruns even while the control is disabled.\n                selected_concat_mode = st.selectbox(\n                    tr("Video Concat Mode"),\n                    options=[VideoConcatMode.sequential.value],\n                    index=0,\n                    key="video_concat_mode_cartoon_locked",\n                    format_func=lambda value: dict(\n                        (v, label) for label, v in video_concat_modes\n                    )[value],\n                    disabled=True,\n                    help="AI Cartoon uses one authored narration timeline in sequential order.",\n                )\n            else:\n                selected_concat_mode = stable_selectbox(\n                    tr("Video Concat Mode"),\n                    options=[value for _, value in video_concat_modes],\n                    default_value=_saved_ui_choice(\n                        "video_concat_mode",\n                        [value for _, value in video_concat_modes],\n                        VideoConcatMode.random.value,\n                    ),\n                    key="video_concat_mode_select",\n                    format_func=lambda value: dict(\n                        (v, label) for label, v in video_concat_modes\n                    )[value],\n                    disabled=bool(\n                        st.session_state.get("match_materials_to_script", False)\n                        or st.session_state.get("strict_scene_matching", False)\n                    ),\n                )\n            params.video_concat_mode = VideoConcatMode(selected_concat_mode)\n')

replace_once('app/services/cartoon_engine.py', 'import subprocess\nimport tempfile\nimport textwrap\nfrom dataclasses import dataclass\n', 'import subprocess\nimport sys\nimport tempfile\nimport textwrap\nfrom array import array\nfrom dataclasses import dataclass\n')

replace_once('app/services/cartoon_engine.py', 'def generate_mouth_cues(\n    audio_file: str,\n    mode: Literal["auto", "rhubarb", "heuristic"] = "auto",\n) -> tuple[list[MouthCue], str]:\n    if mode == "heuristic":\n        return [], "heuristic"\n', 'def _heuristic_audio_mouth_cues(audio_file: str) -> list[MouthCue]:\n    """Build deterministic mouth activity from local narration amplitude.\n\n    This is intentionally not a phoneme recognizer. It prevents the fallback mouth\n    cycle from flapping through silence while keeping a lively mix of mouth shapes.\n    Rhubarb remains the higher-precision optional path.\n    """\n    ffmpeg = utils.get_ffmpeg_binary()\n    command = [\n        ffmpeg,\n        "-v",\n        "error",\n        "-i",\n        str(audio_file),\n        "-ac",\n        "1",\n        "-ar",\n        "16000",\n        "-f",\n        "s16le",\n        "-",\n    ]\n    try:\n        completed = subprocess.run(\n            command,\n            capture_output=True,\n            timeout=180,\n            check=False,\n        )\n    except (OSError, subprocess.TimeoutExpired) as exc:\n        logger.warning(f"heuristic lip-sync audio analysis unavailable: {exc}")\n        return []\n    if completed.returncode != 0 or not completed.stdout:\n        detail = completed.stderr.decode("utf-8", errors="replace")[-300:]\n        logger.warning(\n            "heuristic lip-sync audio analysis failed"\n            + (f": {detail}" if detail else "")\n        )\n        return []\n\n    samples = array("h")\n    samples.frombytes(completed.stdout)\n    if sys.byteorder != "little":\n        samples.byteswap()\n    if not samples:\n        return []\n\n    sample_rate = 16000\n    window_samples = int(sample_rate * 0.09)\n    energies: list[float] = []\n    bounds: list[tuple[int, int]] = []\n    for start in range(0, len(samples), window_samples):\n        end = min(len(samples), start + window_samples)\n        if end <= start:\n            continue\n        total = 0.0\n        for sample in samples[start:end]:\n            total += float(sample) * float(sample)\n        rms = math.sqrt(total / (end - start)) / 32768.0\n        energies.append(rms)\n        bounds.append((start, end))\n    if not energies:\n        return []\n\n    ordered = sorted(energies)\n\n    def percentile(fraction: float) -> float:\n        index = int(round((len(ordered) - 1) * fraction))\n        return ordered[max(0, min(len(ordered) - 1, index))]\n\n    noise_floor = percentile(0.20)\n    speech_level = max(percentile(0.82), noise_floor + 1e-6)\n    threshold = max(0.0055, noise_floor * 1.9, speech_level * 0.16)\n    shape_cycle = "ACBEDFGH"\n    cues: list[MouthCue] = []\n    for index, (energy, (start, end)) in enumerate(zip(energies, bounds)):\n        start_t = start / sample_rate\n        end_t = end / sample_rate\n        if energy <= threshold:\n            shape = "X"\n        else:\n            strength = min(\n                1.0,\n                max(0.0, (energy - threshold) / max(1e-6, speech_level - threshold)),\n            )\n            offset = 2 if strength > 0.72 else (1 if strength > 0.38 else 0)\n            shape = shape_cycle[(index + offset) % len(shape_cycle)]\n        cues.append(MouthCue(start_t, end_t, shape))\n    return cues\n\n\ndef generate_mouth_cues(\n    audio_file: str,\n    mode: Literal["auto", "rhubarb", "heuristic"] = "auto",\n) -> tuple[list[MouthCue], str]:\n    if mode == "heuristic":\n        return _heuristic_audio_mouth_cues(audio_file), "heuristic"\n')

replace_once('app/services/cartoon_engine.py', '        return [], "heuristic"\n    with tempfile.TemporaryDirectory(prefix="mpt-rhubarb-") as temp_dir:\n', '        return _heuristic_audio_mouth_cues(audio_file), "heuristic"\n    with tempfile.TemporaryDirectory(prefix="mpt-rhubarb-") as temp_dir:\n')

replace_once('app/services/cartoon_engine.py', '            logger.warning(f"Rhubarb unavailable; falling back to heuristic lip sync: {exc}")\n            return [], "heuristic"\n', '            logger.warning(f"Rhubarb unavailable; falling back to heuristic lip sync: {exc}")\n            return _heuristic_audio_mouth_cues(audio_file), "heuristic"\n')

replace_once('app/services/cartoon_engine.py', '            logger.warning(f"Rhubarb failed; falling back to heuristic lip sync: {detail}")\n            return [], "heuristic"\n', '            logger.warning(f"Rhubarb failed; falling back to heuristic lip sync: {detail}")\n            return _heuristic_audio_mouth_cues(audio_file), "heuristic"\n')

replace_once('app/services/cartoon_engine.py', '            if mode == "rhubarb":\n                raise CartoonRenderError(f"invalid Rhubarb output: {exc}") from exc\n            return [], "heuristic"\n        cues = _parse_rhubarb_payload(payload)\n        return (cues, "rhubarb") if cues else ([], "heuristic")\n', '            if mode == "rhubarb":\n                raise CartoonRenderError(f"invalid Rhubarb output: {exc}") from exc\n            return _heuristic_audio_mouth_cues(audio_file), "heuristic"\n        cues = _parse_rhubarb_payload(payload)\n        if cues:\n            return cues, "rhubarb"\n        return _heuristic_audio_mouth_cues(audio_file), "heuristic"\n')

replace_once('app/services/cartoon_engine.py', '    for cue in cues:\n        if cue.start <= t < cue.end:\n            return cue.value\n    # Fast deterministic fallback. It is intentionally irregular rather than a simple\n', '    for cue in cues:\n        if cue.start <= t < cue.end:\n            return cue.value\n    if cues:\n        # Audio-derived cues explicitly encode quiet windows. When no cue covers this\n        # timestamp, close the mouth instead of reviving the time-only fallback cycle.\n        return "X"\n    # Fast deterministic fallback. It is intentionally irregular rather than a simple\n')

replace_once('app/services/cartoon_engine.py', '        bob = self.sy(math.sin(t * 4.2 + (0.8 if facing < 0 else 0)) * 4)\n        y = base_y + bob\n        outline = self.palette.ink\n', '        identity_host = color == self.palette.host\n        phase = 0.0 if identity_host else 0.85\n        bob = self.sy(math.sin(t * 3.4 + phase) * 6)\n        entrance = _ease_out_cubic(min(1.0, local_progress * 8.0))\n        nod = self.sy(math.sin(t * 2.1 + phase) * 2.5)\n        y = base_y + bob + nod + self.sy((1.0 - entrance) * 28)\n        outline = self.palette.ink\n')

replace_once('app/services/cartoon_engine.py', '        draw.ellipse(\n            (x - body_w / 2, y, x + body_w / 2, y + body_h),\n            fill=alpha_color,\n            outline=outline,\n            width=self.sc(9 * scale),\n        )\n        head_y = y - self.sy(105 * scale)\n', '        draw.rounded_rectangle(\n            (x - body_w / 2, y, x + body_w / 2, y + body_h),\n            radius=max(self.sc(28), self.sc(82 * scale)),\n            fill=alpha_color,\n            outline=outline,\n            width=self.sc(9 * scale),\n        )\n        # Stable clothing details make the recurring host/guest read as characters\n        # rather than interchangeable colored blobs.\n        collar_y = y + self.sy(48 * scale)\n        draw.polygon(\n            (\n                (x - self.sx(58 * scale), collar_y),\n                (x, collar_y + self.sy(52 * scale)),\n                (x + self.sx(58 * scale), collar_y),\n                (x, collar_y + self.sy(108 * scale)),\n            ),\n            fill=shadow,\n        )\n        badge_x = x - self.sx(62 * scale) if identity_host else x + self.sx(62 * scale)\n        draw.ellipse(\n            (\n                badge_x - self.sx(14 * scale),\n                y + self.sy(155 * scale),\n                badge_x + self.sx(14 * scale),\n                y + self.sy(183 * scale),\n            ),\n            fill=self.palette.accent if identity_host else self.palette.cyan,\n            outline=outline,\n            width=max(1, self.sc(3 * scale)),\n        )\n        head_y = y - self.sy(105 * scale)\n')

replace_once('app/services/cartoon_engine.py', '        draw.ellipse(\n            (x - head_r, head_y - head_r, x + head_r, head_y + head_r),\n            fill=color,\n            outline=outline,\n            width=self.sc(9 * scale),\n        )\n        # Small asymmetric cheek patch gives the characters an authored, non-template feel.\n', '        # Ears sit behind the face and help the head feel constructed rather than stamped.\n        ear_r = self.sx(23 * scale)\n        for ear_x in (x - head_r, x + head_r):\n            draw.ellipse(\n                (\n                    ear_x - ear_r,\n                    head_y - ear_r * 0.15,\n                    ear_x + ear_r,\n                    head_y + ear_r * 1.85,\n                ),\n                fill=shadow,\n                outline=outline,\n                width=max(1, self.sc(5 * scale)),\n            )\n        draw.ellipse(\n            (x - head_r, head_y - head_r, x + head_r, head_y + head_r),\n            fill=color,\n            outline=outline,\n            width=self.sc(9 * scale),\n        )\n        # Distinct silhouettes: the host has a three-point quiff, the guest a side sweep.\n        hair_fill = "#342A38" if identity_host else "#302E55"\n        if identity_host:\n            draw.polygon(\n                (\n                    (x - self.sx(72 * scale), head_y - self.sy(102 * scale)),\n                    (x - self.sx(28 * scale), head_y - self.sy(145 * scale)),\n                    (x + self.sx(2 * scale), head_y - self.sy(106 * scale)),\n                    (x + self.sx(42 * scale), head_y - self.sy(142 * scale)),\n                    (x + self.sx(78 * scale), head_y - self.sy(98 * scale)),\n                ),\n                fill=hair_fill,\n            )\n        else:\n            draw.pieslice(\n                (\n                    x - self.sx(108 * scale),\n                    head_y - self.sy(128 * scale),\n                    x + self.sx(108 * scale),\n                    head_y + self.sy(18 * scale),\n                ),\n                188,\n                352,\n                fill=hair_fill,\n            )\n            draw.ellipse(\n                (\n                    x + self.sx(68 * scale),\n                    head_y - self.sy(62 * scale),\n                    x + self.sx(113 * scale),\n                    head_y + self.sy(22 * scale),\n                ),\n                fill=hair_fill,\n            )\n        # Small asymmetric cheek patch gives the characters an authored, non-template feel.\n')

replace_once('app/services/cartoon_engine.py', '        else:\n            for eye_x in (x - eye_gap, x + eye_gap):\n                draw.ellipse(\n                    (eye_x - eye_r, eye_y - eye_r, eye_x + eye_r, eye_y + eye_r),\n                    fill=outline,\n                )\n', '        else:\n            pupil_shift = facing * self.sc(2 * scale)\n            for eye_x in (x - eye_gap, x + eye_gap):\n                white_r = self.sc(13 * scale)\n                draw.ellipse(\n                    (\n                        eye_x - white_r,\n                        eye_y - white_r,\n                        eye_x + white_r,\n                        eye_y + white_r,\n                    ),\n                    fill="#F8F5F0",\n                    outline=outline,\n                    width=max(1, self.sc(3 * scale)),\n                )\n                pupil_r = self.sc(6 * scale)\n                draw.ellipse(\n                    (\n                        eye_x + pupil_shift - pupil_r,\n                        eye_y - pupil_r,\n                        eye_x + pupil_shift + pupil_r,\n                        eye_y + pupil_r,\n                    ),\n                    fill=outline,\n                )\n')

replace_once('app/services/cartoon_engine.py', '        self._mouth(draw, x, head_y + self.sy(62 * scale), mouth, scale)\n\n        shoulder_y = y + self.sy(105 * scale)\n        arm_length = self.sx(150 * scale)\n        gesture = _ease_out_cubic(min(1.0, local_progress * 3.0))\n', '        # Tiny nose mark keeps expressions readable at vertical-video scale.\n        nose_x = x + facing * self.sx(8 * scale)\n        draw.arc(\n            (\n                nose_x - self.sx(12 * scale),\n                head_y + self.sy(10 * scale),\n                nose_x + self.sx(16 * scale),\n                head_y + self.sy(42 * scale),\n            ),\n            20 if facing > 0 else 160,\n            205 if facing > 0 else 345,\n            fill=shadow,\n            width=max(1, self.sc(4 * scale)),\n        )\n        self._mouth(draw, x, head_y + self.sy(62 * scale), mouth, scale)\n\n        shoulder_y = y + self.sy(105 * scale)\n        arm_length = self.sx(150 * scale)\n        gesture_base = _ease_out_cubic(min(1.0, local_progress * 3.0))\n        emphasis = 0.5 + 0.5 * math.sin(t * 2.8 + phase)\n        gesture = min(1.0, gesture_base * (0.84 + 0.16 * emphasis))\n')

replace_once('app/services/cartoon_engine.py', '        elif action == "write":\n            hand_x = x + facing * self.sx(115 * scale)\n            hand_y = self.sy(1345)\n            start = right_shoulder if facing > 0 else left_shoulder\n            draw.line((start[0], start[1], hand_x, hand_y), fill=outline, width=self.sc(13 * scale))\n            draw.line((hand_x, hand_y, hand_x + facing * self.sx(55 * scale), hand_y - self.sy(25 * scale)), fill=self.palette.accent, width=self.sc(7 * scale))\n        else:\n', '        elif action == "write":\n            hand_x = x + facing * self.sx(115 * scale)\n            hand_y = self.sy(1345)\n            start = right_shoulder if facing > 0 else left_shoulder\n            draw.line((start[0], start[1], hand_x, hand_y), fill=outline, width=self.sc(13 * scale))\n            draw.line((hand_x, hand_y, hand_x + facing * self.sx(55 * scale), hand_y - self.sy(25 * scale)), fill=self.palette.accent, width=self.sc(7 * scale))\n        elif action == "react":\n            lift = self.sy((24 + 22 * emphasis) * scale)\n            hand_x = x + facing * self.sx(118 * scale)\n            hand_y = shoulder_y - lift\n            start = right_shoulder if facing > 0 else left_shoulder\n            draw.line((start[0], start[1], hand_x, hand_y), fill=outline, width=self.sc(12 * scale))\n            draw.ellipse(\n                (\n                    hand_x - self.sc(13 * scale),\n                    hand_y - self.sc(13 * scale),\n                    hand_x + self.sc(13 * scale),\n                    hand_y + self.sc(13 * scale),\n                ),\n                fill=color,\n                outline=outline,\n                width=max(1, self.sc(4 * scale)),\n            )\n            other = left_shoulder if facing > 0 else right_shoulder\n            draw.line(\n                (\n                    other[0],\n                    other[1],\n                    other[0] - facing * self.sx(38 * scale),\n                    other[1] + self.sy(115 * scale),\n                ),\n                fill=outline,\n                width=self.sc(12 * scale),\n            )\n        else:\n')

replace_once('app/services/cartoon_engine.py', '    def render(self, t: float) -> Image.Image:\n', '    def _compact_overlay(\n        self,\n        image: Image.Image,\n        scene: CartoonScene,\n        progress: float,\n        t: float,\n    ) -> None:\n        """Reuse the full explanatory graphic as a picture-in-picture concept card."""\n        if scene.overlay == "none":\n            return\n        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))\n        self._overlay(ImageDraw.Draw(layer), scene, progress, t)\n        source = (\n            self.sx(85),\n            self.sy(285),\n            self.sx(995),\n            self.sy(1215),\n        )\n        card = layer.crop(source)\n        max_width = self.sx(820)\n        max_height = self.sy(600)\n        factor = min(max_width / card.width, max_height / card.height)\n        size = (\n            max(1, int(round(card.width * factor))),\n            max(1, int(round(card.height * factor))),\n        )\n        card = card.resize(size, Image.Resampling.LANCZOS)\n        left = (self.width - card.width) // 2\n        top = self.sy(62)\n        image.paste(card, (left, top), card)\n\n    def render(self, t: float) -> Image.Image:\n')

replace_once('app/services/cartoon_engine.py', '        if scene.layout == "two_shot":\n            self._microphone(draw, self.sx(390), self.sy(1180), flip=False)\n            self._microphone(draw, self.sx(690), self.sy(1180), flip=True)\n', '        if scene.layout == "two_shot":\n            has_overlay = scene.overlay != "none"\n            pair_base_y = self.sy(1040 if has_overlay else 980)\n            pair_scale = 0.84 if has_overlay else 0.95\n            self._microphone(\n                draw,\n                self.sx(390),\n                self.sy(1215 if has_overlay else 1180),\n                flip=False,\n            )\n            self._microphone(\n                draw,\n                self.sx(690),\n                self.sy(1215 if has_overlay else 1180),\n                flip=True,\n            )\n')

replace_once('app/services/cartoon_engine.py', '                x=self.sx(285),\n                base_y=self.sy(980),\n', '                x=self.sx(285),\n                base_y=pair_base_y,\n')

replace_once('app/services/cartoon_engine.py', '                scale=0.95,\n                dimmed=not speaking_host,\n', '                scale=pair_scale,\n                dimmed=not speaking_host,\n')

replace_once('app/services/cartoon_engine.py', '                x=self.sx(795),\n                base_y=self.sy(980),\n', '                x=self.sx(795),\n                base_y=pair_base_y,\n')

replace_once('app/services/cartoon_engine.py', '                scale=0.95,\n                dimmed=speaking_host,\n', '                scale=pair_scale,\n                dimmed=speaking_host,\n')

replace_once('app/services/cartoon_engine.py', '            color = self.palette.host if host_layout else self.palette.guest\n            shadow = self.palette.host_shadow if host_layout else self.palette.guest_shadow\n            facing = 1 if host_layout else -1\n            self._microphone(draw, self.sx(660 if host_layout else 420), self.sy(1085), flip=not host_layout)\n            self._character(\n                draw,\n                x=self.sx(470 if host_layout else 610),\n                base_y=self.sy(860),\n', '            color = self.palette.host if host_layout else self.palette.guest\n            shadow = self.palette.host_shadow if host_layout else self.palette.guest_shadow\n            facing = 1 if host_layout else -1\n            has_overlay = scene.overlay != "none"\n            presenter_base_y = self.sy(1040 if has_overlay else 860)\n            presenter_scale = 0.96 if has_overlay else 1.26\n            self._microphone(\n                draw,\n                self.sx(660 if host_layout else 420),\n                self.sy(1210 if has_overlay else 1085),\n                flip=not host_layout,\n            )\n            self._character(\n                draw,\n                x=self.sx(470 if host_layout else 610),\n                base_y=presenter_base_y,\n')

replace_once('app/services/cartoon_engine.py', '                scale=1.26,\n            )\n\n        # Non-graphic shots can still carry a compact explanatory card in the upper area.\n        if scene.overlay != "none":\n            card_progress = _ease_out_cubic(min(1.0, progress * 4.0))\n            card_w = self.sx(720 * card_progress)\n            card_h = self.sy(210)\n            cx = self.width // 2\n            top = self.sy(100)\n            if card_w > self.sx(30):\n                draw.rounded_rectangle((cx - card_w / 2, top, cx + card_w / 2, top + card_h), radius=self.sc(40), fill=self.palette.card, outline=self.palette.ink, width=self.sc(7))\n                if card_w > self.sx(420):\n                    lines = _wrap_lines(scene.overlay_text or scene.accent_text, width=24, max_lines=2)\n                    font = _font(self.sc(44))\n                    y = top + self.sy(42)\n                    for line in lines:\n                        _text_centered(draw, (cx, y), line, font, self.palette.card_ink)\n                        y += self.sy(58)\n        return image\n', '                scale=presenter_scale,\n            )\n\n        # Presenter and two-shot layouts keep the recurring cast prominent while the\n        # actual semantic visual (browser, email, checklist, brain, etc.) appears above.\n        if scene.overlay != "none":\n            self._compact_overlay(image, scene, progress, t)\n        return image\n')


test_path = Path("test/services/test_cartoon_engine.py")
test_text = test_path.read_text(encoding="utf-8")
test_marker = "def test_audio_aware_heuristic_mouth_closes_between_cues():"
if test_marker not in test_text:
    test_text += """


def test_audio_aware_heuristic_mouth_closes_between_cues():
    cues = [cartoon_engine.MouthCue(0.0, 0.1, "A")]
    assert cartoon_engine._mouth_at(cues, 0.05, True) == "A"
    assert cartoon_engine._mouth_at(cues, 0.20, True) == "X"
    assert cartoon_engine._mouth_at(cues, 0.05, False) == "X"


def test_generate_heuristic_mouth_cues_uses_audio_activity():
    expected = [
        cartoon_engine.MouthCue(0.0, 0.09, "X"),
        cartoon_engine.MouthCue(0.09, 0.18, "C"),
    ]
    with patch(
        "app.services.cartoon_engine._heuristic_audio_mouth_cues",
        return_value=expected,
    ) as analyze:
        cues, backend = cartoon_engine.generate_mouth_cues(
            "C:/task/audio.mp3",
            mode="heuristic",
        )
    assert backend == "heuristic"
    assert cues == expected
    analyze.assert_called_once_with("C:/task/audio.mp3")


def test_non_graphic_semantic_overlay_renders_real_visual_card():
    scene = cartoon_engine.CartoonScene(
        scene=1,
        start=0.0,
        end=4.0,
        narration="Keep the browser tab open.",
        layout="host",
        speaker="host",
        emotion="thinking",
        action="point",
        overlay="browser_tabs",
        overlay_text="OPEN TABS",
    )
    renderer = cartoon_engine.DoodleRenderer(
        width=360,
        height=640,
        scenes=[scene],
        mouth_cues=[],
    )
    frame = renderer.render(1.0)
    upper = frame.crop((0, 0, 360, 290))
    light_pixels = sum(
        1
        for red, green, blue in upper.getdata()
        if red > 205 and green > 205 and blue > 205
    )
    # A real browser concept card occupies a substantial part of the upper frame;
    # the old implementation only drew a shallow title strip.
    assert light_pixels > 18000
"""
    test_path.write_text(test_text, encoding="utf-8")

contract_path = Path("test/services/test_cartoon_webui_contract.py")
if not contract_path.exists():
    contract_path.write_text(
        """from pathlib import Path


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
""",
        encoding="utf-8",
    )
