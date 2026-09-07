# Local Semantic Ranker

Optional local OpenCLIP service for `Strict Scene Matching`.

The service ranks stock-video candidates by comparing the scene query against provider preview images. It is intentionally isolated from the main MoneyPrinterTurbo environment so OpenCLIP/PyTorch cannot interfere with Chatterbox or the application's locked dependencies.

## Windows setup

```powershell
cd C:\Projects\LocalShortsStudio\semantic_ranker
.\setup.ps1
```

The setup creates `semantic_ranker\.venv`, installs OpenCLIP 3.3.0, then installs the CUDA 12.4 PyTorch 2.6.0 / torchvision 0.21.0 pair already validated on the target RTX 4060 laptop.

## Start

```powershell
cd C:\Projects\LocalShortsStudio\semantic_ranker
.\start.ps1
```

The service listens only on `http://127.0.0.1:4124`.

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:4124/health
```

Default model:

- model: `ViT-B-32`
- pretrained: `laion2b_s34b_b79k`
- device: `auto` (CUDA when available, otherwise CPU)

Optional environment overrides:

```powershell
$env:SEMANTIC_RANKER_DEVICE = "cuda"
$env:SEMANTIC_RANKER_MODEL = "ViT-B-32"
$env:SEMANTIC_RANKER_PRETRAINED = "laion2b_s34b_b79k"
.\start.ps1
```

## Ranking strategy

For each provider query the service:

1. scores one preview from every candidate;
2. keeps the strongest five candidates;
3. scores up to three additional previews for those candidates;
4. averages the two strongest preview similarities per candidate;
5. returns candidates sorted from most to least semantically relevant.

This reduces preview downloads while being more robust than trusting a single thumbnail.

Only HTTPS preview URLs hosted by Pexels, Pixabay, or Coverr domains are accepted. Redirects are validated again after download.

If the service is unavailable, LocalShortsStudio keeps the existing provider ordering and Strict Scene Matching continues without semantic reranking.

OpenCLIP is consumed as a dependency and is MIT licensed; no OpenCLIP source code is vendored into this repository.
