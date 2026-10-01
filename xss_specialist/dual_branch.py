"""Dual-branch XSS structural model.

Two independent encoders are initialized from the successful v0.4 BERT checkpoint:
- semantic branch: source, sink, defense
- flow branch: connected/disconnected/unknown relation

The branches never share trainable parameters, preventing cross-task gradient
interference. Neither branch is a final vulnerability judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn
from transformers import AutoModel

from xss_specialist.multitask import (
    IGNORE_INDEX,
    SOURCE_LABELS,
    SINK_LABELS,
    DEFENSE_LABELS,
    FLOW_LABELS,
)

SEMANTIC_LABEL_SPACES = {
    "source": SOURCE_LABELS,
    "sink": SINK_LABELS,
    "defense": DEFENSE_LABELS,
}
LABEL_SPACES = {**SEMANTIC_LABEL_SPACES, "flow": FLOW_LABELS}


@dataclass
class BranchOutput:
    loss: torch.Tensor | None
    logits: dict[str, torch.Tensor]


def _mlp(hidden: int, classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.LayerNorm(hidden),
        nn.Dropout(0.1),
        nn.Linear(hidden, classes),
    )


def _pooled(encoded):
    pooled = getattr(encoded, "pooler_output", None)
    if pooled is None:
        pooled = encoded.last_hidden_state[:, 0]
    return pooled


class XSSDualBranchModel(nn.Module):
    def __init__(self, backbone_name_or_path: str):
        super().__init__()
        self.backbone_name_or_path = backbone_name_or_path

        self.semantic_backbone = AutoModel.from_pretrained(backbone_name_or_path)
        self.flow_backbone = AutoModel.from_pretrained(backbone_name_or_path)

        sem_hidden = int(self.semantic_backbone.config.hidden_size)
        flow_hidden = int(self.flow_backbone.config.hidden_size)

        self.semantic_dropout = nn.Dropout(0.1)
        self.flow_dropout = nn.Dropout(0.1)

        self.semantic_heads = nn.ModuleDict({
            name: _mlp(sem_hidden, len(labels))
            for name, labels in SEMANTIC_LABEL_SPACES.items()
        })
        self.flow_head = _mlp(flow_hidden, len(FLOW_LABELS))

        self.semantic_task_weights = {
            "source": 0.5,
            "sink": 2.0,
            "defense": 2.5,
        }
        self.class_weights: dict[str, torch.Tensor | None] = {
            name: None for name in LABEL_SPACES
        }

    def configure_losses(
        self,
        *,
        semantic_task_weights: dict[str, float] | None = None,
        class_weights: dict[str, torch.Tensor | None] | None = None,
    ) -> None:
        if semantic_task_weights:
            for name, value in semantic_task_weights.items():
                if name not in SEMANTIC_LABEL_SPACES:
                    raise ValueError(f"unknown semantic task: {name}")
                self.semantic_task_weights[name] = float(value)
        if class_weights:
            for name, value in class_weights.items():
                if name not in LABEL_SPACES:
                    raise ValueError(f"unknown task: {name}")
                self.class_weights[name] = value

    @staticmethod
    def _backbone_kwargs(input_ids, attention_mask, token_type_ids):
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            out["token_type_ids"] = token_type_ids
        return out

    def forward_semantic(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        source_labels: torch.Tensor | None = None,
        sink_labels: torch.Tensor | None = None,
        defense_labels: torch.Tensor | None = None,
    ) -> BranchOutput:
        encoded = self.semantic_backbone(**self._backbone_kwargs(
            input_ids, attention_mask, token_type_ids
        ))
        pooled = self.semantic_dropout(_pooled(encoded))
        logits = {
            name: head(pooled)
            for name, head in self.semantic_heads.items()
        }
        targets = {
            "source": source_labels,
            "sink": sink_labels,
            "defense": defense_labels,
        }

        losses = []
        weights = []
        for name, target in targets.items():
            if target is None:
                continue
            active = target.ne(IGNORE_INDEX)
            if not active.any():
                continue
            class_weight = self.class_weights.get(name)
            if class_weight is not None:
                class_weight = class_weight.to(logits[name].device)
            task_loss = nn.functional.cross_entropy(
                logits[name],
                target,
                weight=class_weight,
                ignore_index=IGNORE_INDEX,
            )
            task_weight = float(self.semantic_task_weights.get(name, 1.0))
            losses.append(task_loss * task_weight)
            weights.append(task_weight)

        loss = (
            torch.stack(losses).sum() / max(sum(weights), 1e-12)
            if losses else None
        )
        return BranchOutput(loss=loss, logits=logits)

    def forward_flow(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        flow_labels: torch.Tensor | None = None,
    ) -> BranchOutput:
        encoded = self.flow_backbone(**self._backbone_kwargs(
            input_ids, attention_mask, token_type_ids
        ))
        pooled = self.flow_dropout(_pooled(encoded))
        logits = {"flow": self.flow_head(pooled)}

        loss = None
        if flow_labels is not None and flow_labels.ne(IGNORE_INDEX).any():
            class_weight = self.class_weights.get("flow")
            if class_weight is not None:
                class_weight = class_weight.to(logits["flow"].device)
            loss = nn.functional.cross_entropy(
                logits["flow"],
                flow_labels,
                weight=class_weight,
                ignore_index=IGNORE_INDEX,
            )
        return BranchOutput(loss=loss, logits=logits)

    def semantic_parameters(self):
        yield from self.semantic_backbone.parameters()
        yield from self.semantic_heads.parameters()

    def flow_parameters(self):
        yield from self.flow_backbone.parameters()
        yield from self.flow_head.parameters()

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.semantic_backbone.save_pretrained(output / "semantic_backbone")
        self.flow_backbone.save_pretrained(output / "flow_backbone")
        torch.save(self.semantic_heads.state_dict(), output / "semantic_heads.pt")
        torch.save(self.flow_head.state_dict(), output / "flow_head.pt")
        (output / "dual_branch_config.json").write_text(
            json.dumps({
                "schema": "xss-dual-branch-v2",
                "semantic_backbone": "semantic_backbone",
                "flow_backbone": "flow_backbone",
                "semantic_tasks": list(SEMANTIC_LABEL_SPACES),
                "flow_task": "flow",
                "label_spaces": LABEL_SPACES,
                "branches_share_trainable_parameters": False,
                "final_judge": False,
                "confirmation_authority": "deterministic_or_browser_execution_only",
            }, indent=2) + "\n"
        )

    @classmethod
    def load(cls, output_dir: str | Path, map_location: str = "cpu"):
        output = Path(output_dir)
        model = cls(str(output / "semantic_backbone"))
        model.flow_backbone = AutoModel.from_pretrained(output / "flow_backbone")
        sem = torch.load(
            output / "semantic_heads.pt",
            map_location=map_location,
            weights_only=True,
        )
        flow = torch.load(
            output / "flow_head.pt",
            map_location=map_location,
            weights_only=True,
        )
        model.semantic_heads.load_state_dict(sem)
        model.flow_head.load_state_dict(flow)
        return model


class XSSFlowBranchModel(nn.Module):
    """Standalone flow branch initialized from the v0.4-compatible encoder."""

    def __init__(self, backbone_name_or_path: str):
        super().__init__()
        self.backbone_name_or_path = backbone_name_or_path
        self.backbone = AutoModel.from_pretrained(backbone_name_or_path)
        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.head = _mlp(hidden, len(FLOW_LABELS))
        self.class_weights: torch.Tensor | None = None

    def configure_class_weights(self, weights: torch.Tensor | None) -> None:
        self.class_weights = weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        flow_labels: torch.Tensor | None = None,
    ) -> BranchOutput:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        encoded = self.backbone(**kwargs)
        pooled = self.dropout(_pooled(encoded))
        logits = {"flow": self.head(pooled)}

        loss = None
        if flow_labels is not None and flow_labels.ne(IGNORE_INDEX).any():
            weight = self.class_weights
            if weight is not None:
                weight = weight.to(logits["flow"].device)
            loss = nn.functional.cross_entropy(
                logits["flow"],
                flow_labels,
                weight=weight,
                ignore_index=IGNORE_INDEX,
            )
        return BranchOutput(loss=loss, logits=logits)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output / "flow_backbone")
        torch.save(self.head.state_dict(), output / "flow_head.pt")
        (output / "flow_branch_config.json").write_text(
            json.dumps({
                "schema": "xss-flow-branch-v2",
                "flow_backbone": "flow_backbone",
                "labels": FLOW_LABELS,
                "final_judge": False,
                "confirmation_authority": "deterministic_or_browser_execution_only",
            }, indent=2) + "\n"
        )

    @classmethod
    def load(cls, output_dir: str | Path, map_location: str = "cpu"):
        output = Path(output_dir)
        model = cls(str(output / "flow_backbone"))
        state = torch.load(
            output / "flow_head.pt",
            map_location=map_location,
            weights_only=True,
        )
        model.head.load_state_dict(state)
        return model
