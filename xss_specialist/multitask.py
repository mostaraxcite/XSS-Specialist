"""Direct-PyTorch multi-task XSS specialist.

The model predicts independent structural properties instead of a whole-snippet
SAFE/XSS label. Missing task labels are masked with IGNORE_INDEX so mixed
corpora can supervise only the facts they actually contain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

IGNORE_INDEX = -100

SOURCE_LABELS = ["NONE", "BROWSER", "SERVER", "FRAMEWORK", "OTHER"]
SINK_LABELS = [
    "NONE",
    "DANGEROUS_HTML",
    "DANGEROUS_JS",
    "DANGEROUS_URL",
    "SAFE_OUTPUT",
    "OTHER",
]
DEFENSE_LABELS = [
    "NONE",
    "SANITIZATION",
    "CONTEXTUAL_ENCODING",
    "FRAMEWORK_ESCAPING",
    "SAFE_DOM_API",
    "INPUT_CONSTRAINT",
    "OTHER",
]
FLOW_LABELS = ["CONNECTED", "DISCONNECTED", "UNKNOWN"]

LABEL_SPACES = {
    "source": SOURCE_LABELS,
    "sink": SINK_LABELS,
    "defense": DEFENSE_LABELS,
    "flow": FLOW_LABELS,
}


@dataclass
class MultiTaskOutput:
    loss: torch.Tensor | None
    logits: dict[str, torch.Tensor]


class XSSMultiTaskModel(nn.Module):
    def __init__(self, backbone_name_or_path: str):
        super().__init__()
        self.backbone_name_or_path = backbone_name_or_path
        self.backbone = AutoModel.from_pretrained(backbone_name_or_path)
        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(0.1),
                nn.Linear(hidden, len(labels)),
            )
            for name, labels in LABEL_SPACES.items()
        })
        self.task_loss_weights = {name: 1.0 for name in LABEL_SPACES}
        self.class_weights: dict[str, torch.Tensor | None] = {
            name: None for name in LABEL_SPACES
        }

    def configure_losses(
        self,
        *,
        task_loss_weights: dict[str, float] | None = None,
        class_weights: dict[str, torch.Tensor | None] | None = None,
    ) -> None:
        if task_loss_weights:
            for name, value in task_loss_weights.items():
                if name not in LABEL_SPACES:
                    raise ValueError(f"unknown task: {name}")
                self.task_loss_weights[name] = float(value)
        if class_weights:
            for name, value in class_weights.items():
                if name not in LABEL_SPACES:
                    raise ValueError(f"unknown task: {name}")
                self.class_weights[name] = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        source_labels: torch.Tensor | None = None,
        sink_labels: torch.Tensor | None = None,
        defense_labels: torch.Tensor | None = None,
        flow_labels: torch.Tensor | None = None,
    ) -> MultiTaskOutput:
        backbone_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids
        encoded = self.backbone(**backbone_kwargs)
        pooled = getattr(encoded, "pooler_output", None)
        if pooled is None:
            pooled = encoded.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        logits = {name: head(pooled) for name, head in self.heads.items()}
        labels = {
            "source": source_labels,
            "sink": sink_labels,
            "defense": defense_labels,
            "flow": flow_labels,
        }

        losses = []
        loss_weights = []
        for name, target in labels.items():
            if target is None:
                continue
            active = target.ne(IGNORE_INDEX)
            if active.any():
                class_weight = self.class_weights.get(name)
                if class_weight is not None:
                    class_weight = class_weight.to(logits[name].device)
                task_loss = nn.functional.cross_entropy(
                    logits[name],
                    target,
                    weight=class_weight,
                    ignore_index=IGNORE_INDEX,
                )
                task_weight = float(self.task_loss_weights.get(name, 1.0))
                losses.append(task_loss * task_weight)
                loss_weights.append(task_weight)

        loss = (
            torch.stack(losses).sum() / max(sum(loss_weights), 1e-12)
            if losses else None
        )
        return MultiTaskOutput(loss=loss, logits=logits)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output / "backbone")
        torch.save(self.heads.state_dict(), output / "heads.pt")
        (output / "multitask_config.json").write_text(
            json.dumps({
                "schema": "xss-multitask-v1",
                "backbone": "backbone",
                "label_spaces": LABEL_SPACES,
                "head_type": "two_layer_mlp",
                "final_judge": False,
            }, indent=2) + "\n"
        )

    @classmethod
    def load(cls, output_dir: str | Path, map_location: str = "cpu") -> "XSSMultiTaskModel":
        output = Path(output_dir)
        model = cls(str(output / "backbone"))
        state = torch.load(output / "heads.pt", map_location=map_location, weights_only=True)
        model.heads.load_state_dict(state)
        return model
