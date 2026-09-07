"""Local OpenCLIP service used to rerank stock-video preview images.

This process is intentionally separate from MoneyPrinterTurbo so OpenCLIP/PyTorch
cannot disturb the main application environment or the Chatterbox environment.
"""

from __future__ import annotations

import io
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager, nullcontext
from urllib.parse import urljoin, urlsplit

import requests
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_PORT = 4124
MAX_CANDIDATES = 20
MAX_PREVIEWS_PER_CANDIDATE = 4
MAX_NEGATIVE_QUERIES = 8
MAX_REFERENCE_PREVIEWS = 12
DEEP_RERANK_TOP_K = 5
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REDIRECTS = 3
EMBEDDING_CACHE_SIZE = 512
NEGATIVE_WEIGHT = 0.45
DIVERSITY_WEIGHT = 0.35
DIVERSITY_THRESHOLD = 0.80
ALLOWED_PREVIEW_HOST_SUFFIXES = (
    "pexels.com",
    "pixabay.com",
    "coverr.co",
)

_model_lock = threading.Lock()
_model_bundle = None
_model_error = None
_embedding_cache_lock = threading.Lock()
_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
_metrics_lock = threading.Lock()
_rank_requests_total = 0
_align_requests_total = 0
_last_rank_query = ""
_last_rank_candidates = 0
_last_align_scenes = 0
_last_align_terms = 0


class Candidate(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    preview_urls: list[str] = Field(
        default_factory=list,
        max_length=MAX_PREVIEWS_PER_CANDIDATE,
    )


class AlignRequest(BaseModel):
    scenes: list[str] = Field(min_length=1, max_length=64)
    terms: list[str] = Field(min_length=1, max_length=32)


class RankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    negative_queries: list[str] = Field(default_factory=list, max_length=MAX_NEGATIVE_QUERIES)
    reference_preview_urls: list[str] = Field(
        default_factory=list,
        max_length=MAX_REFERENCE_PREVIEWS,
    )
    candidates: list[Candidate] = Field(min_length=1, max_length=MAX_CANDIDATES)


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return min(maximum, max(minimum, value))


def _negative_weight() -> float:
    return _float_env("SEMANTIC_RANKER_NEGATIVE_WEIGHT", NEGATIVE_WEIGHT, 0.0, 2.0)


def _diversity_weight() -> float:
    return _float_env("SEMANTIC_RANKER_DIVERSITY_WEIGHT", DIVERSITY_WEIGHT, 0.0, 2.0)


def _diversity_threshold() -> float:
    return _float_env(
        "SEMANTIC_RANKER_DIVERSITY_THRESHOLD",
        DIVERSITY_THRESHOLD,
        -1.0,
        1.0,
    )


def _configured_device(torch_module) -> str:
    configured = str(os.environ.get("SEMANTIC_RANKER_DEVICE", "auto") or "auto").strip().lower()
    if configured == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("SEMANTIC_RANKER_DEVICE=cuda but CUDA is not available")
    if configured not in {"cpu", "cuda"}:
        raise RuntimeError("SEMANTIC_RANKER_DEVICE must be auto, cpu, or cuda")
    return configured


def _load_model():
    global _model_bundle, _model_error
    if _model_bundle is not None:
        return _model_bundle

    with _model_lock:
        if _model_bundle is not None:
            return _model_bundle
        try:
            import open_clip
            import torch

            model_name = str(os.environ.get("SEMANTIC_RANKER_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL)
            pretrained = str(
                os.environ.get("SEMANTIC_RANKER_PRETRAINED", DEFAULT_PRETRAINED)
                or DEFAULT_PRETRAINED
            )
            device = _configured_device(torch)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model = model.to(device)
            model.eval()
            _model_bundle = {
                "torch": torch,
                "model": model,
                "preprocess": preprocess,
                "tokenizer": tokenizer,
                "device": device,
                "model_name": model_name,
                "pretrained": pretrained,
            }
            _model_error = None
        except Exception as exc:
            _model_error = f"{type(exc).__name__}: {exc}"
            raise
    return _model_bundle


def _allowed_preview_url(url: str) -> bool:
    try:
        parsed = urlsplit(str(url or "").strip())
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    hostname = parsed.hostname.lower().rstrip(".")
    return any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in ALLOWED_PREVIEW_HOST_SUFFIXES
    )


def _read_bounded_image_response(response: requests.Response) -> Image.Image:
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except (TypeError, ValueError):
            declared_size = 0
        if declared_size > MAX_IMAGE_BYTES:
            raise ValueError("preview image exceeds size limit")

    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise ValueError("preview image exceeds size limit")
        chunks.append(chunk)

    try:
        image = Image.open(io.BytesIO(b"".join(chunks)))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"preview is not a decodable image: {type(exc).__name__}") from exc
    return image.convert("RGB")


def _fetch_image(url: str) -> Image.Image:
    """Fetch a provider preview while validating every redirect before following it."""
    current_url = str(url or "").strip()
    if not _allowed_preview_url(current_url):
        raise ValueError("preview URL is not an allowed HTTPS stock-provider host")

    session = requests.Session()
    try:
        for redirect_index in range(MAX_REDIRECTS + 1):
            with session.get(
                current_url,
                timeout=(5, 20),
                allow_redirects=False,
                stream=True,
                headers={"User-Agent": "LocalShortsStudio-SemanticRanker/1.1"},
            ) as response:
                if response.is_redirect or response.is_permanent_redirect:
                    if redirect_index >= MAX_REDIRECTS:
                        raise ValueError("preview exceeded redirect limit")
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("preview redirect is missing Location")
                    next_url = urljoin(current_url, location)
                    if not _allowed_preview_url(next_url):
                        raise ValueError(
                            "preview redirect left the allowed stock-provider hosts"
                        )
                    current_url = next_url
                    continue

                return _read_bounded_image_response(response)
    finally:
        session.close()

    raise ValueError("preview could not be fetched")


def _autocast_context(bundle):
    torch = bundle["torch"]
    if bundle["device"] == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _encode_images(images: list[Image.Image]) -> list[list[float]]:
    if not images:
        return []
    bundle = _load_model()
    torch = bundle["torch"]
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    device = bundle["device"]

    image_tensor = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.inference_mode(), _autocast_context(bundle):
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.detach().float().cpu().tolist()


def _encode_texts(texts: list[str]) -> list[list[float]]:
    normalized = [str(text or "").strip() for text in texts if str(text or "").strip()]
    if not normalized:
        return []
    bundle = _load_model()
    torch = bundle["torch"]
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]

    text_tensor = tokenizer(normalized).to(device)
    with torch.inference_mode(), _autocast_context(bundle):
        text_features = model.encode_text(text_tensor)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.detach().float().cpu().tolist()


def _cache_get(url: str) -> list[float] | None:
    with _embedding_cache_lock:
        feature = _embedding_cache.get(url)
        if feature is None:
            return None
        _embedding_cache.move_to_end(url)
        return feature


def _cache_put(url: str, feature: list[float]) -> None:
    with _embedding_cache_lock:
        _embedding_cache[url] = feature
        _embedding_cache.move_to_end(url)
        while len(_embedding_cache) > EMBEDDING_CACHE_SIZE:
            _embedding_cache.popitem(last=False)


def _image_features_for_urls(urls: list[str]) -> dict[str, list[float]]:
    """Fetch and encode only cache misses, returning normalized CPU embeddings."""
    normalized_urls: list[str] = []
    seen: set[str] = set()
    for value in urls:
        url = str(value or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        normalized_urls.append(url)

    result: dict[str, list[float]] = {}
    missing: list[str] = []
    for url in normalized_urls:
        feature = _cache_get(url)
        if feature is None:
            missing.append(url)
        else:
            result[url] = feature

    images: list[Image.Image] = []
    fetched_urls: list[str] = []
    for url in missing:
        try:
            images.append(_fetch_image(url))
            fetched_urls.append(url)
        except Exception:
            continue

    for url, feature in zip(fetched_urls, _encode_images(images)):
        _cache_put(url, feature)
        result[url] = feature
    return result


def _dot(first: list[float], second: list[float]) -> float:
    return sum(left * right for left, right in zip(first, second))


def _aggregate(scores: list[float]) -> float:
    """Reward candidates that match across more than a single lucky preview."""
    if not scores:
        return -1.0
    strongest = sorted(scores, reverse=True)[: min(2, len(scores))]
    return sum(strongest) / len(strongest)


def _final_score(
    positive_score: float,
    negative_score: float,
    diversity_similarity: float,
    *,
    negative_weight: float | None = None,
    diversity_weight: float | None = None,
    diversity_threshold: float | None = None,
) -> float:
    """Combine scene relevance, anti-concepts, and repetition penalty."""
    if positive_score <= -1.0:
        return -1.0
    neg_weight = _negative_weight() if negative_weight is None else negative_weight
    div_weight = _diversity_weight() if diversity_weight is None else diversity_weight
    threshold = (
        _diversity_threshold() if diversity_threshold is None else diversity_threshold
    )
    repetition = max(0.0, diversity_similarity - threshold)
    return positive_score - (neg_weight * max(0.0, negative_score)) - (
        div_weight * repetition
    )


def _candidate_metrics(
    image_features: list[list[float]],
    positive_feature: list[float],
    negative_features: list[list[float]],
    reference_features: list[list[float]],
) -> dict[str, float]:
    if not image_features:
        return {
            "score": -1.0,
            "positive_score": -1.0,
            "negative_score": 0.0,
            "diversity_similarity": 0.0,
        }

    positive_scores = [_dot(feature, positive_feature) for feature in image_features]
    positive_score = _aggregate(positive_scores)

    negative_score = 0.0
    if negative_features:
        negative_scores = [
            max(_dot(feature, negative) for negative in negative_features)
            for feature in image_features
        ]
        negative_score = _aggregate(negative_scores)

    diversity_similarity = 0.0
    if reference_features:
        diversity_similarity = max(
            _dot(feature, reference)
            for feature in image_features
            for reference in reference_features
        )

    return {
        "score": _final_score(
            positive_score,
            negative_score,
            diversity_similarity,
        ),
        "positive_score": positive_score,
        "negative_score": negative_score,
        "diversity_similarity": diversity_similarity,
    }



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


def _rank(request: RankRequest) -> dict:
    bundle = _load_model()
    positive_features = _encode_texts([request.query])
    if not positive_features:
        raise ValueError("query could not be encoded")
    positive_feature = positive_features[0]
    negative_features = _encode_texts(request.negative_queries)
    reference_features = list(
        _image_features_for_urls(request.reference_preview_urls).values()
    )

    per_candidate_features: dict[str, list[list[float]]] = {
        candidate.id: [] for candidate in request.candidates
    }

    # Stage 1: one preview for every provider candidate. Cached image embeddings
    # make repeated scene-by-scene reranking cheap after the first encounter.
    stage1_urls = [
        candidate.preview_urls[0]
        for candidate in request.candidates
        if candidate.preview_urls
    ]
    stage1_feature_map = _image_features_for_urls(stage1_urls)
    for candidate in request.candidates:
        if not candidate.preview_urls:
            continue
        feature = stage1_feature_map.get(candidate.preview_urls[0])
        if feature is not None:
            per_candidate_features[candidate.id].append(feature)

    top_ids = [
        candidate_id
        for candidate_id, _ in sorted(
            (
                (
                    candidate_id,
                    _candidate_metrics(
                        features,
                        positive_feature,
                        negative_features,
                        reference_features,
                    )["score"],
                )
                for candidate_id, features in per_candidate_features.items()
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:DEEP_RERANK_TOP_K]
    ]
    top_id_set = set(top_ids)

    # Stage 2: inspect extra previews only for the strongest candidates.
    stage2_urls: list[str] = []
    for candidate in request.candidates:
        if candidate.id in top_id_set:
            stage2_urls.extend(candidate.preview_urls[1:])
    stage2_feature_map = _image_features_for_urls(stage2_urls)
    for candidate in request.candidates:
        if candidate.id not in top_id_set:
            continue
        for preview_url in candidate.preview_urls[1:]:
            feature = stage2_feature_map.get(preview_url)
            if feature is not None:
                per_candidate_features[candidate.id].append(feature)

    ranked = []
    for candidate in request.candidates:
        metrics = _candidate_metrics(
            per_candidate_features[candidate.id],
            positive_feature,
            negative_features,
            reference_features,
        )
        ranked.append(
            {
                "id": candidate.id,
                "score": round(metrics["score"], 6),
                "positive_score": round(metrics["positive_score"], 6),
                "negative_score": round(metrics["negative_score"], 6),
                "diversity_similarity": round(
                    metrics["diversity_similarity"],
                    6,
                ),
                "preview_count": len(per_candidate_features[candidate.id]),
            }
        )
    ranked.sort(key=lambda row: row["score"], reverse=True)

    return {
        "model": f"{bundle['model_name']}/{bundle['pretrained']}",
        "device": bundle["device"],
        "negative_queries": list(request.negative_queries),
        "reference_count": len(reference_features),
        "scoring": {
            "negative_weight": _negative_weight(),
            "diversity_weight": _diversity_weight(),
            "diversity_threshold": _diversity_threshold(),
        },
        "ranked": ranked,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    _load_model()
    yield


app = FastAPI(
    title="LocalShortsStudio Semantic Ranker",
    version="1.2",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    bundle = _model_bundle
    with _embedding_cache_lock:
        cache_entries = len(_embedding_cache)
    with _metrics_lock:
        metrics = {
            "rank_requests_total": _rank_requests_total,
            "align_requests_total": _align_requests_total,
            "last_rank_query": _last_rank_query,
            "last_rank_candidates": _last_rank_candidates,
            "last_align_scenes": _last_align_scenes,
            "last_align_terms": _last_align_terms,
        }
    return {
        "status": "healthy" if bundle is not None else "initializing",
        "model_loaded": bundle is not None,
        "model": (
            f"{bundle['model_name']}/{bundle['pretrained']}" if bundle is not None else None
        ),
        "device": bundle.get("device") if bundle is not None else None,
        "version": "1.2",
        "negative_weight": _negative_weight(),
        "diversity_weight": _diversity_weight(),
        "diversity_threshold": _diversity_threshold(),
        "embedding_cache_entries": cache_entries,
        **metrics,
        "error": _model_error,
    }



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


@app.post("/rank")
def rank(request: RankRequest):
    global _rank_requests_total, _last_rank_query, _last_rank_candidates
    with _metrics_lock:
        _rank_requests_total += 1
        _last_rank_query = request.query
        _last_rank_candidates = len(request.candidates)
    try:
        return _rank(request)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"semantic ranking failed: {type(exc).__name__}: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SEMANTIC_RANKER_PORT", DEFAULT_PORT))
    uvicorn.run(app, host="127.0.0.1", port=port)
