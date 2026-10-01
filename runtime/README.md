# XSS Specialist Runtime v1

Standalone runtime bundle for the frozen XSS Specialist release.

Included:
- Semantic model: Source Head v4e winner
- Flow model: Flow v2 winner
- Deterministic semantic/flow authority
- Browser oracle and scope-bounded live components already present in the project

Excluded:
- training datasets
- tuning scripts
- External v6/v7 row-level data
- failed checkpoints

## Install

```bash
python -m pip install uv
uv sync
uv run playwright install chromium
```

## Verify runtime

```bash
uv run python -m runtime.xss_specialist_runtime doctor
```

## Run model field inference

```bash
uv run python -m runtime.xss_specialist_runtime fields \
  --source "location.hash" \
  --sink "element.innerHTML" \
  --statement "element.innerHTML = location.hash"
```

Frozen components:
- Semantic run: 36920030570
- Semantic artifact: xss-source-v4e-lr25e5-e12
- Flow run: 36867388302
- Flow artifact: xss-flow-v2-candidate-f0-lr3e5-e8
- External v7 run: 36922054415
- External v7: 90% system / 90% model advice
