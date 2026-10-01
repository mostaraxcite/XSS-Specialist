"""Adapter registry for the XSS specialist.

The registry is the single source of truth for which XSS classifier is
"current". It abstracts over two adapter shapes:

  - **full**: a complete Hugging Face model directory (e.g. v0.4 bert-tiny,
    saved as `BertForSequenceClassification`).
  - **peft**: a frozen base model identified by name plus a PEFT LoRA adapter
    directory (e.g. xss-v0.5 = MiniLM-L6 + a PEFT LoRA adapter).

Inference code asks the registry for a classifier pipeline and never has to
care which kind it gets. Rollback to a previous adapter is a registry swap, not
a code change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "security-models" / "xss" / "adapters" / "REGISTRY.json"


AdapterKind = Literal["full", "peft"]


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    kind: AdapterKind
    base_model: str
    path: str
    description: str = ""
    promoted: bool = False
    metrics: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "AdapterSpec":
        return cls(
            name=raw["name"],
            kind=raw["kind"],
            base_model=raw["base_model"],
            path=raw["path"],
            description=raw.get("description", ""),
            promoted=bool(raw.get("promoted", False)),
            metrics=dict(raw.get("metrics", {})),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "base_model": self.base_model,
            "path": self.path,
            "description": self.description,
            "promoted": self.promoted,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class Registry:
    registry_path: Path
    adapters: dict[str, AdapterSpec]
    active: str

    def get(self, name: str | None = None) -> AdapterSpec:
        target = name or self.active
        if target not in self.adapters:
            raise KeyError(
                f"Adapter '{target}' not in registry. Known: {sorted(self.adapters)}"
            )
        return self.adapters[target]

    def list_adapters(self) -> list[AdapterSpec]:
        return [self.adapters[name] for name in sorted(self.adapters)]

    def with_active(self, name: str) -> "Registry":
        if name not in self.adapters:
            raise KeyError(f"Unknown adapter '{name}'")
        return Registry(self.registry_path, self.adapters, name)

    def write(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": self.active,
            "adapters": {name: spec.to_dict() for name, spec in self.adapters.items()},
        }
        self.registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve(path_root: str) -> Path:
    raw = Path(path_root)
    return raw if raw.is_absolute() else (ROOT / raw).resolve()


def load_registry(path: Path | None = None) -> Registry:
    registry_path = path or DEFAULT_REGISTRY_PATH
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    adapters = {name: AdapterSpec.from_dict(payload) for name, payload in raw["adapters"].items()}
    return Registry(registry_path=registry_path, adapters=adapters, active=raw["active"])


def default_registry() -> Registry:
    if DEFAULT_REGISTRY_PATH.exists():
        return load_registry(DEFAULT_REGISTRY_PATH)
    # First-run fallback: the legacy v0.4 model directory counts as a "full"
    # adapter. This keeps backward compatibility when the registry has not been
    # initialised yet.
    fallback = Registry(
        registry_path=DEFAULT_REGISTRY_PATH,
        adapters={
            "v0.4-baseline": AdapterSpec(
                name="v0.4-baseline",
                kind="full",
                base_model="google/bert_uncased_L-4_H-256_A-4",
                path="security-models/xss/model",
                description="Full bert-tiny fine-tune, the v0.4 promotion gate.",
                promoted=True,
            )
        },
        active="v0.4-baseline",
    )
    return fallback


def write_registry(registry: Registry) -> None:
    registry.write()


def load_classifier(
    name: str | None = None,
    registry: Registry | None = None,
    device: int = -1,
) -> tuple[object, AdapterSpec]:
    """Return (transformers.pipeline, AdapterSpec) for the requested adapter.

    Loading is deterministic: full adapters are loaded directly; PEFT adapters
    are composed on top of their frozen base model.
    """
    reg = registry or default_registry()
    spec = reg.get(name)
    resolved = _resolve(spec.path)

    if spec.kind == "full":
        if not resolved.exists():
            raise FileNotFoundError(
                f"Promoted adapter '{spec.name}' expects local model files at {resolved}, "
                "but they are not bundled in Git. Reproduce the v0.4 model using "
                "security-models/xss/README.md before default inference, or explicitly "
                "select a bundled research adapter with --adapter for evaluation."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(resolved))
        model = AutoModelForSequenceClassification.from_pretrained(str(resolved))
    elif spec.kind == "peft":
        if not resolved.exists():
            raise FileNotFoundError(
                f"PEFT adapter '{spec.name}' is missing at {resolved}. "
                "The registry will never fall back to an untrained base classifier."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(resolved))
        base = AutoModelForSequenceClassification.from_pretrained(
            spec.base_model,
            num_labels=3,
            id2label={0: "SAFE", 1: "POSSIBLE_XSS", 2: "XSS"},
            label2id={"SAFE": 0, "POSSIBLE_XSS": 1, "XSS": 2},
        )
        try:
            model = PeftModel.from_pretrained(base, str(resolved), is_trainable=False)
        except (KeyError, RuntimeError, ValueError) as exc:
            # Never fall back to a base model with a freshly initialised head: that would serve
            # random predictions under the adapter's name.
            raise RuntimeError(
                f"Adapter '{spec.name}' at {resolved} could not be applied to {spec.base_model} "
                "with a 3-label head; it must include the trained classifier (modules_to_save)."
            ) from exc
    else:
        raise ValueError(f"Unknown adapter kind: {spec.kind}")

    model.eval()
    if device == -1:
        model.to("cpu")
    classify = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=device,
    )
    return classify, spec


def initial_registry_with_v04() -> Registry:
    """Build the canonical starting registry: v0.4 baseline + xss-v0.5 stub."""
    reg = default_registry()
    if "xss-v0.5" not in reg.adapters:
        reg = Registry(
            registry_path=reg.registry_path,
            adapters={
                **reg.adapters,
                "xss-v0.5": AdapterSpec(
                    name="xss-v0.5",
                    kind="peft",
                    base_model="nreimers/MiniLM-L6-H384-uncased",
                    path="security-models/xss/adapters/xss-v05",
                    description="MiniLM-L6 PEFT candidate; see adapter metadata for historical targets.",
                    promoted=False,
                ),
            },
            active=reg.active,
        )
        reg.write()
    return reg


def register_adapter(
    registry: Registry,
    *,
    name: str,
    kind: AdapterKind,
    base_model: str,
    path: str,
    description: str = "",
    promoted: bool = False,
    metrics: dict | None = None,
    activate: bool = False,
) -> Registry:
    spec = AdapterSpec(
        name=name,
        kind=kind,
        base_model=base_model,
        path=path,
        description=description,
        promoted=promoted,
        metrics=dict(metrics or {}),
    )
    adapters = {**registry.adapters, name: spec}
    new_active = name if activate else registry.active
    new_registry = Registry(registry.registry_path, adapters, new_active)
    new_registry.write()
    return new_registry
