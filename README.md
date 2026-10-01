# XSS Specialist

Standalone runtime repository for the validated XSS Specialist.

## Frozen runtime

- Semantic model: **Source Head v4e winner**
- Sink macro F1: **98.61%**
- Defense macro F1: **91.56%**
- Flow model: **Flow v2**
- External v7: **90% system / 90% model advice**
- Pre-Live gate: **PASS**
- Browser confirmation accuracy: **93.75%**

The model cannot declare an XSS finding confirmed by confidence alone. Browser execution remains the confirmation authority.

## Install

```bash
git clone https://github.com/mostaraxcite/XSS-Specialist.git
cd XSS-Specialist

python -m pip install uv
uv sync
uv run playwright install chromium
```

## Verify

```bash
uv run python -m runtime.xss_specialist_runtime doctor
```

## Test the specialist model

```bash
uv run python -m runtime.xss_specialist_runtime fields \
  --source "location.hash" \
  --sink "element.innerHTML" \
  --statement "element.innerHTML = location.hash"
```

Expected structural result includes:

```text
Source = BROWSER
Sink   = DANGEROUS_HTML
Flow   = CONNECTED
Confirmed = false
```

## Local browser acceptance

```bash
uv run python -m live.evaluate
```

## Release

Current verified release:

**v1.0.1**

https://github.com/mostaraxcite/XSS-Specialist/releases/tag/v1.0.1

The release ZIP includes the model weights and runtime code.

## Repository policy

This is a runtime repository. It intentionally excludes:

- training datasets
- tuning scripts
- failed checkpoints
- External v6 row-level cases
- External v7 row-level cases

Live findings remain review/quarantine evidence and are not automatically fed back into model training.
