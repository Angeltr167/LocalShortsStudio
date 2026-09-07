from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    assert old in text, f"marker not found in {path}: {old[:120]!r}"
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Expose exact narration timing to material selection.
replace_once(
    "app/services/task.py",
    """    downloaded_videos = get_video_materials(\n        task_id,\n        params,\n        video_terms,\n        audio_duration,\n        loomloom_video_request=loomloom_video_request,\n    )\n""",
    """    downloaded_videos = get_video_materials(\n        task_id,\n        params,\n        video_terms,\n        audio_duration,\n        subtitle_path=subtitle_path,\n        loomloom_video_request=loomloom_video_request,\n    )\n""",
)
replace_once(
    "app/services/task.py",
    """def get_video_materials(\n    task_id,\n    params,\n    video_terms,\n    audio_duration,\n    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,\n):\n""",
    """def get_video_materials(\n    task_id,\n    params,\n    video_terms,\n    audio_duration,\n    subtitle_path: str = \"\",\n    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,\n):\n""",
)
replace_once(
    "app/services/task.py",
    """                semantic_scene_ranking=_semantic_scene_ranking_enabled(params),\n            )\n""",
    """                semantic_scene_ranking=_semantic_scene_ranking_enabled(params),\n                narration_subtitle_path=(\n                    subtitle_path if _semantic_scene_ranking_enabled(params) else \"\"\n                ),\n            )\n""",
)

# 2) Thread subtitle timing through the stock material downloader.
material_path = Path("app/services/material.py")
material_text = material_path.read_text(encoding="utf-8")
old_sig = """    strict_scene_matching: bool = False,\n    semantic_scene_ranking: bool = False,\n) -> List[str]:\n"""
new_sig = """    strict_scene_matching: bool = False,\n    semantic_scene_ranking: bool = False,\n    narration_subtitle_path: str = \"\",\n) -> List[str]:\n"""
assert old_sig in material_text, "material downloader signature marker not found"
material_text = material_text.replace(old_sig, new_sig, 1)
old_call = """            semantic_scene_ranking=semantic_scene_ranking,\n        )\n"""
new_call = """            semantic_scene_ranking=semantic_scene_ranking,\n            narration_subtitle_path=narration_subtitle_path,\n        )\n"""
assert old_call in material_text, "strict scene call marker not found"
material_text = material_text.replace(old_call, new_call, 1)
material_path.write_text(material_text, encoding="utf-8")

# 3) Add text-to-text alignment client to the main app.
semantic_path = Path("app/services/semantic_ranker.py")
semantic_text = semantic_path.read_text(encoding="utf-8")
insert_marker = """def rank_materials(\n"""
align_client = r'''
def align_scene_terms(
    scene_texts: list[str],
    terms: list[str],
    *,
    enabled: bool,
) -> list[list[float]] | None:
    """Ask the local OpenCLIP service for scene-text -> visual-term similarities.

    This is fail-open. Strict Scene Matching keeps its deterministic positional
    assignment if the ranker is disabled, offline, or returns an invalid matrix.
    """
    normalized_scenes = [" ".join(str(value or "").split()) for value in scene_texts]
    normalized_terms = [" ".join(str(value or "").split()) for value in terms]
    if (
        not enabled
        or not normalized_scenes
        or not normalized_terms
        or not any(normalized_scenes)
    ):
        return None

    base_url = _base_url()
    try:
        session = _local_session(base_url)
        try:
            response = session.post(
                f"{base_url}/align",
                json={"scenes": normalized_scenes, "terms": normalized_terms},
                timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
            )
        finally:
            session.close()
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        logger.warning(
            "semantic narration alignment unavailable, keep positional scene plan: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return None

    scores = body.get("scores") if isinstance(body, dict) else None
    if not isinstance(scores, list) or len(scores) != len(normalized_scenes):
        logger.warning("semantic narration alignment returned an invalid matrix")
        return None

    matrix: list[list[float]] = []
    for row in scores:
        if not isinstance(row, list) or len(row) != len(normalized_terms):
            return None
        normalized_row: list[float] = []
        for value in row:
            try:
                score = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(score):
                return None
            normalized_row.append(score)
        matrix.append(normalized_row)
    return matrix


'''
assert insert_marker in semantic_text, "semantic ranker insert marker missing"
semantic_text = semantic_text.replace(insert_marker, align_client + insert_marker, 1)
semantic_path.write_text(semantic_text, encoding="utf-8")

# 4) Make strict scene planning narration-aware using already-generated SRT timings.
strict_path = Path("app/services/strict_scene.py")
strict_text = strict_path.read_text(encoding="utf-8")
strict_text = strict_text.replace("import math\n", "import math\nimport re\n", 1)
strict_text = strict_text.replace(
    "from app.services import semantic_ranker, task_artifacts\n",
    "from app.services import semantic_ranker, subtitle, task_artifacts\n",
    1,
)

marker = """def _fallback_query_indexes(primary_index: int, query_count: int) -> list[int]:\n"""
helpers = r'''
_SRT_TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+),(?P<ms>\d+)")
_MIN_MEANINGFUL_TAIL_SECONDS = 1.0
_POSITION_PRIOR_WEIGHT = 0.08


def _srt_time_seconds(value: str) -> float | None:
    match = _SRT_TIME_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    parts = {key: int(raw) for key, raw in match.groupdict().items()}
    return (
        parts["h"] * 3600
        + parts["m"] * 60
        + parts["s"]
        + parts["ms"] / 1000.0
    )


def _subtitle_cues(subtitle_path: str) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    if not subtitle_path:
        return cues
    for _, times, text in subtitle.file_to_subtitles(subtitle_path):
        try:
            start_raw, end_raw = [part.strip() for part in times.split(" --> ", 1)]
        except ValueError:
            continue
        start = _srt_time_seconds(start_raw)
        end = _srt_time_seconds(end_raw)
        narration = " ".join(str(text or "").split())
        if start is None or end is None or end <= start or not narration:
            continue
        cues.append((start, end, narration))
    return cues


def _attach_narration_text(
    scene_plan: list[dict[str, Any]],
    subtitle_path: str,
) -> list[str]:
    cues = _subtitle_cues(subtitle_path)
    scene_texts: list[str] = []
    for scene in scene_plan:
        start = float(scene["start"])
        end = float(scene["end"])
        overlapping = [
            text
            for cue_start, cue_end, text in cues
            if cue_end > start and cue_start < end
        ]
        narration = " ".join(overlapping)
        scene["narration_text"] = narration
        scene_texts.append(narration)
    return scene_texts


def _monotonic_alignment(
    scores: list[list[float]],
    scene_plan: list[dict[str, Any]],
    term_count: int,
) -> list[int] | None:
    """Choose a nondecreasing scene->term path with real closing-term exposure."""
    scene_count = len(scene_plan)
    if (
        scene_count == 0
        or term_count <= 0
        or len(scores) != scene_count
        or scene_count < term_count
    ):
        return None
    if any(len(row) != term_count for row in scores):
        return None

    meaningful_end = scene_count - 1
    if (
        float(scene_plan[-1].get("duration") or 0) < _MIN_MEANINGFUL_TAIL_SECONDS
        and scene_count > 1
    ):
        meaningful_end -= 1
    active_count = meaningful_end + 1
    if active_count < term_count:
        return None

    neg_inf = float("-inf")
    dp = [[neg_inf] * term_count for _ in range(active_count)]
    prev = [[-1] * term_count for _ in range(active_count)]

    def weighted_score(scene_index: int, term_index: int) -> float:
        reference_duration = max(float(scene_plan[0].get("duration") or 1), 0.001)
        duration_weight = max(
            0.15,
            float(scene_plan[scene_index].get("duration") or 0) / reference_duration,
        )
        if active_count <= 1 or term_count <= 1:
            position_prior = 1.0
        else:
            scene_position = scene_index / (active_count - 1)
            term_position = term_index / (term_count - 1)
            position_prior = 1.0 - abs(scene_position - term_position)
        return (scores[scene_index][term_index] * duration_weight) + (
            _POSITION_PRIOR_WEIGHT * position_prior
        )

    dp[0][0] = weighted_score(0, 0)
    for scene_index in range(1, active_count):
        for term_index in range(term_count):
            for previous_term in (term_index, term_index - 1):
                if previous_term < 0 or dp[scene_index - 1][previous_term] == neg_inf:
                    continue
                candidate = dp[scene_index - 1][previous_term] + weighted_score(
                    scene_index, term_index
                )
                if candidate > dp[scene_index][term_index]:
                    dp[scene_index][term_index] = candidate
                    prev[scene_index][term_index] = previous_term

    if dp[meaningful_end][term_count - 1] == neg_inf:
        return None

    alignment = [0] * active_count
    current = term_count - 1
    for scene_index in range(meaningful_end, -1, -1):
        alignment[scene_index] = current
        if scene_index > 0:
            current = prev[scene_index][current]
            if current < 0:
                return None

    alignment.extend([term_count - 1] * (scene_count - active_count))
    return alignment


def _apply_narration_alignment(
    scene_plan: list[dict[str, Any]],
    terms: list[str],
    subtitle_path: str,
    *,
    enabled: bool,
) -> bool:
    scene_texts = _attach_narration_text(scene_plan, subtitle_path)
    similarity = semantic_ranker.align_scene_terms(
        scene_texts,
        terms,
        enabled=enabled,
    )
    if similarity is None:
        return False
    alignment = _monotonic_alignment(similarity, scene_plan, len(terms))
    if alignment is None:
        return False

    for scene, term_index, row in zip(scene_plan, alignment, similarity):
        scene["query_index"] = term_index
        scene["query"] = terms[term_index]
        scene["narration_aligned"] = True
        scene["narration_alignment_score"] = round(float(row[term_index]), 6)
    return True


'''
assert marker in strict_text, "strict helper marker missing"
strict_text = strict_text.replace(marker, helpers + marker, 1)

old_bridge = '''    # When audio needs more timeline slots than there are visual intents, the\n    # deterministic distribution creates adjacent repeats. Mark only the final\n    # slot of a repeated run as a semantic bridge toward the next narrative\n    # intent. Strict mode without semantic ranking keeps its existing behavior;\n    # the bridge metadata is consumed only by the semantic downloader path.\n    run_start = 0\n    while run_start < len(plan):\n        run_query_index = int(plan[run_start]["query_index"])\n        run_end = run_start + 1\n        while (\n            run_end < len(plan)\n            and int(plan[run_end]["query_index"]) == run_query_index\n        ):\n            run_end += 1\n\n        if run_end - run_start > 1 and run_end < len(plan):\n            next_query_index = int(plan[run_end]["query_index"])\n            if next_query_index > run_query_index:\n                bridge_scene = plan[run_end - 1]\n                bridge_scene["bridge_to_query_index"] = next_query_index\n                bridge_scene["bridge_to_query"] = terms[next_query_index]\n                bridge_scene["bridge_query"] = _build_bridge_query(\n                    terms[run_query_index],\n                    terms[next_query_index],\n                )\n\n        run_start = run_end\n\n    return plan\n'''
assert old_bridge in strict_text, "old inline bridge planner block missing"
strict_text = strict_text.replace(old_bridge, "    return plan\n", 1)

bridge_helper_marker = """def _build_bridge_query(\n"""
bridge_helper = r'''
def _mark_semantic_bridges(
    plan: list[dict[str, Any]],
    terms: list[str],
) -> None:
    for scene in plan:
        for key in ("bridge_to_query_index", "bridge_to_query", "bridge_query"):
            scene.pop(key, None)

    run_start = 0
    while run_start < len(plan):
        run_query_index = int(plan[run_start]["query_index"])
        run_end = run_start + 1
        while (
            run_end < len(plan)
            and int(plan[run_end]["query_index"]) == run_query_index
        ):
            run_end += 1

        if run_end - run_start > 1 and run_end < len(plan):
            next_query_index = int(plan[run_end]["query_index"])
            if next_query_index > run_query_index:
                bridge_scene = plan[run_end - 1]
                bridge_scene["bridge_to_query_index"] = next_query_index
                bridge_scene["bridge_to_query"] = terms[next_query_index]
                bridge_scene["bridge_query"] = _build_bridge_query(
                    terms[run_query_index],
                    terms[next_query_index],
                )
        run_start = run_end


'''
assert bridge_helper_marker in strict_text, "bridge query helper marker missing"
strict_text = strict_text.replace(bridge_helper_marker, bridge_helper + bridge_helper_marker, 1)

strict_text = strict_text.replace(
    """    semantic_scene_ranking: bool = False,\n) -> List[str]:\n""",
    """    semantic_scene_ranking: bool = False,\n    narration_subtitle_path: str = \"\",\n) -> List[str]:\n""",
    1,
)
strict_text = strict_text.replace(
    """    if not scene_plan:\n        logger.warning(\"strict scene matching could not build a scene plan\")\n        return []\n\n    task_artifacts.patch_script_data(task_id, scene_plan=scene_plan)\n""",
    """    if not scene_plan:\n        logger.warning(\"strict scene matching could not build a scene plan\")\n        return []\n\n    narration_aligned = _apply_narration_alignment(\n        scene_plan,\n        terms,\n        narration_subtitle_path,\n        enabled=semantic_scene_ranking,\n    )\n    if semantic_scene_ranking:\n        _mark_semantic_bridges(scene_plan, terms)\n    logger.info(\n        \"strict scene narration alignment: \"\n        f\"enabled={semantic_scene_ranking}, applied={narration_aligned}\"\n    )\n\n    task_artifacts.patch_script_data(task_id, scene_plan=scene_plan)\n""",
    1,
)
strict_path.write_text(strict_text, encoding="utf-8")

# 5) Add /align plus explicit request counters to the local OpenCLIP service.
service_path = Path("semantic_ranker/main.py")
service_text = service_path.read_text(encoding="utf-8")
service_text = service_text.replace(
    """_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()\n""",
    """_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()\n_metrics_lock = threading.Lock()\n_rank_requests_total = 0\n_align_requests_total = 0\n_last_rank_query = \"\"\n_last_rank_candidates = 0\n_last_align_scenes = 0\n_last_align_terms = 0\n""",
    1,
)
service_text = service_text.replace(
    """class RankRequest(BaseModel):\n""",
    """class AlignRequest(BaseModel):\n    scenes: list[str] = Field(min_length=1, max_length=64)\n    terms: list[str] = Field(min_length=1, max_length=32)\n\n\nclass RankRequest(BaseModel):\n""",
    1,
)
rank_marker = """def _rank(request: RankRequest) -> dict:\n"""
align_impl = r'''
def _align(request: AlignRequest) -> dict:
    scene_texts = [" ".join(str(value or "").split()) for value in request.scenes]
    term_texts = [" ".join(str(value or "").split()) for value in request.terms]
    if not all(scene_texts) or not all(term_texts):
        raise ValueError("alignment scenes and terms must be non-empty")

    features = _encode_texts(scene_texts + term_texts)
    if len(features) != len(scene_texts) + len(term_texts):
        raise ValueError("alignment text encoding returned an unexpected shape")
    scene_features = features[: len(scene_texts)]
    term_features = features[len(scene_texts) :]
    scores = [
        [round(_dot(scene_feature, term_feature), 6) for term_feature in term_features]
        for scene_feature in scene_features
    ]
    bundle = _load_model()
    return {
        "model": f"{bundle['model_name']}/{bundle['pretrained']}",
        "device": bundle["device"],
        "scores": scores,
    }


'''
assert rank_marker in service_text, "service rank marker missing"
service_text = service_text.replace(rank_marker, align_impl + rank_marker, 1)

old_rank_endpoint = '''@app.post("/rank")\ndef rank(request: RankRequest):\n    try:\n        return _rank(request)\n'''
new_rank_endpoint = '''@app.post("/rank")\ndef rank(request: RankRequest):\n    global _rank_requests_total, _last_rank_query, _last_rank_candidates\n    with _metrics_lock:\n        _rank_requests_total += 1\n        _last_rank_query = request.query\n        _last_rank_candidates = len(request.candidates)\n    try:\n        return _rank(request)\n'''
assert old_rank_endpoint in service_text, "rank endpoint marker missing"
service_text = service_text.replace(old_rank_endpoint, new_rank_endpoint, 1)

endpoint_marker = """@app.post(\"/rank\")\n"""
align_endpoint = r'''
@app.post("/align")
def align(request: AlignRequest):
    global _align_requests_total, _last_align_scenes, _last_align_terms
    with _metrics_lock:
        _align_requests_total += 1
        _last_align_scenes = len(request.scenes)
        _last_align_terms = len(request.terms)
    try:
        return _align(request)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"semantic alignment failed: {type(exc).__name__}: {exc}",
        ) from exc


'''
assert endpoint_marker in service_text, "endpoint insertion marker missing"
service_text = service_text.replace(endpoint_marker, align_endpoint + endpoint_marker, 1)

old_health = '''    with _embedding_cache_lock:\n        cache_entries = len(_embedding_cache)\n    return {\n'''
new_health = '''    with _embedding_cache_lock:\n        cache_entries = len(_embedding_cache)\n    with _metrics_lock:\n        metrics = {\n            "rank_requests_total": _rank_requests_total,\n            "align_requests_total": _align_requests_total,\n            "last_rank_query": _last_rank_query,\n            "last_rank_candidates": _last_rank_candidates,\n            "last_align_scenes": _last_align_scenes,\n            "last_align_terms": _last_align_terms,\n        }\n    return {\n'''
assert old_health in service_text, "health marker missing"
service_text = service_text.replace(old_health, new_health, 1)
service_text = service_text.replace(
    '        "embedding_cache_entries": cache_entries,\n        "error": _model_error,\n',
    '        "embedding_cache_entries": cache_entries,\n        **metrics,\n        "error": _model_error,\n',
    1,
)
service_text = service_text.replace('    version="1.1",\n', '    version="1.2",\n', 1)
service_text = service_text.replace('        "version": "1.1",\n', '        "version": "1.2",\n', 1)
service_path.write_text(service_text, encoding="utf-8")

# 6) Focused tests.
test_path = Path("test/services/test_semantic_scene_ranking.py")
test_text = test_path.read_text(encoding="utf-8")
test_text += r'''


def test_monotonic_narration_alignment_reaches_final_term_before_tiny_tail():
    plan = strict_scene.build_scene_plan(
        ["first", "middle", "closing"],
        audio_duration=9.1,
        max_clip_duration=3,
    )
    scores = [
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.0, 0.0, 0.95],
    ]
    alignment = strict_scene._monotonic_alignment(scores, plan, 3)
    assert alignment == [0, 1, 2, 2]
    assert plan[-1]["duration"] == pytest.approx(0.1)


def test_apply_narration_alignment_uses_real_subtitle_text(tmp_path):
    subtitle_path = tmp_path / "narration.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,900\nphone in bed\n\n"
        "2\n00:00:03,000 --> 00:00:05,900\nwrite the task down\n\n",
        encoding="utf-8",
    )
    plan = strict_scene.build_scene_plan(
        ["checking smartphone", "writing notebook"],
        audio_duration=6,
        max_clip_duration=3,
    )
    with patch(
        "app.services.strict_scene.semantic_ranker.align_scene_terms",
        return_value=[[0.9, 0.1], [0.1, 0.9]],
    ):
        applied = strict_scene._apply_narration_alignment(
            plan,
            ["checking smartphone", "writing notebook"],
            str(subtitle_path),
            enabled=True,
        )
    assert applied is True
    assert [scene["query_index"] for scene in plan] == [0, 1]
    assert plan[0]["narration_text"] == "phone in bed"
    assert plan[1]["narration_text"] == "write the task down"


def test_alignment_service_returns_text_similarity_matrix(monkeypatch):
    fake_features = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    monkeypatch.setattr(ranker_service, "_encode_texts", lambda texts: fake_features)
    monkeypatch.setattr(
        ranker_service,
        "_load_model",
        lambda: {"model_name": "test", "pretrained": "unit", "device": "cpu"},
    )
    result = ranker_service._align(
        ranker_service.AlignRequest(
            scenes=["scene one", "scene two"],
            terms=["term one", "term two"],
        )
    )
    assert result["scores"] == [[1.0, 0.0], [0.0, 1.0]]
'''
test_path.write_text(test_text, encoding="utf-8")
