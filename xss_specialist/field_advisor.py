"""Field-level semantic + canonical flow advisory integration.

The model never confirms XSS. Deterministic taint remains the static authority and
browser execution remains the only confirmation authority.
"""
from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoTokenizer

from xss_specialist.dual_branch import XSSDualBranchModel, XSSFlowBranchModel, LABEL_SPACES
from xss_specialist.flow_relations import canonical_relation_text, deterministic_relation
from xss_specialist.semantic_oracle import classify_source, classify_sink, classify_defense


def _top(logits: torch.Tensor, labels: list[str]) -> dict:
    probs=torch.softmax(logits,dim=-1)[0].detach().cpu().tolist()
    dist={label:float(p) for label,p in zip(labels,probs)}
    label=max(dist,key=dist.get)
    return {"label":label,"confidence":dist[label],"probabilities":dist}


class StructuralFieldAdvisor:
    def __init__(self, semantic_dir: str|Path, flow_dir: str|Path, device: str|None=None):
        self.semantic_dir=Path(semantic_dir); self.flow_dir=Path(flow_dir)
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.semantic_model=XSSDualBranchModel.load(self.semantic_dir)
        self.semantic_model.to(self.device); self.semantic_model.eval()
        self.flow_model=XSSFlowBranchModel.load(self.flow_dir)
        self.flow_model.to(self.device); self.flow_model.eval()
        self.semantic_tokenizer=AutoTokenizer.from_pretrained(self.semantic_dir/"semantic_backbone")
        self.flow_tokenizer=AutoTokenizer.from_pretrained(self.flow_dir/"flow_backbone")

    def _encode(self,tok,text,max_length):
        b=tok(text,return_tensors="pt",truncation=True,max_length=max_length)
        return {k:v.to(self.device) for k,v in b.items()}

    @torch.no_grad()
    def semantic_field(self, task: str, expression: str) -> dict:
        if task not in ("source","sink","defense"):
            raise ValueError(f"unsupported semantic field task: {task}")
        text=f"[{task.upper()}]\n{expression}"
        out=self.semantic_model.forward_semantic(**self._encode(self.semantic_tokenizer,text,96))
        model=_top(out.logits[task],LABEL_SPACES[task])

        oracle_fn={
            "source": classify_source,
            "sink": classify_sink,
            "defense": classify_defense,
        }[task]
        deterministic=oracle_fn(expression)

        # Known explicit APIs/literals are structural facts and do not need a
        # statistical guess. Unrecognized fields remain OTHER; model output is
        # retained only as routing advice.
        label=deterministic["label"]
        confidence=1.0 if label != "OTHER" else float(model["confidence"])
        return {
            "label":label,
            "confidence":confidence,
            "task":task,
            "expression":expression,
            "authority":"deterministic" if label != "OTHER" else "bounded_unknown",
            "deterministic":deterministic,
            "model_advice":model,
            "advisory_only":True,
            "confirmed":False,
        }

    @torch.no_grad()
    def flow(self, relation: dict) -> dict:
        row={"task":"FLOW_RELATION",**relation}
        text=canonical_relation_text(row)
        out=self.flow_model(**self._encode(self.flow_tokenizer,text,128))
        model=_top(out.logits["flow"],LABEL_SPACES["flow"])
        deterministic=deterministic_relation(row)

        # A deterministic CONNECTED/DISCONNECTED local relation wins. For an
        # abstention the structural result stays UNKNOWN; the neural branch can
        # only prioritize review and cannot turn uncertainty into proof.
        relation_label=deterministic["relation"]
        return {
            "label":relation_label,
            "confidence":1.0 if deterministic["authoritative"] else float(model["confidence"]),
            "task":"flow",
            "authority":"deterministic" if deterministic["authoritative"] else "bounded_unknown",
            "deterministic":deterministic,
            "model_advice":model,
            "advisory_only":True,
            "confirmed":False,
        }

    def review_fields(self, *, source_expression: str, sink_expression: str,
                      defense_expression: str, relation: dict) -> dict:
        return {
            "engine":"xss-structural-field-v4",
            "source":self.semantic_field("source",source_expression),
            "sink":self.semantic_field("sink",sink_expression),
            "defense":self.semantic_field("defense",defense_expression),
            "flow":self.flow(relation),
            "final_judge":False,
            "confirmed":False,
            "requires_deterministic_review":True,
            "browser_execution_required_for_confirmation":True,
        }
