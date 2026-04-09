from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from decision_service.dataset import canonical_action
from decision_service.models import DecisionResult, ExAnteDecisionRequest
from feature_store.contracts import EX_ANTE_SCHEMA_VERSION, ExAnteFeatureContract
from mlops.registry import latest_manifest_path, load_manifest


MODEL_NAME = "ex_ante_policy"
CONFIDENCE_THRESHOLD = 0.55


def _load_latest_model() -> tuple[Any | None, dict[str, Any]]:
    manifest_path = latest_manifest_path(MODEL_NAME)
    if manifest_path is None:
        return None, {}
    manifest = load_manifest(manifest_path)
    artifact_path = manifest.get("artifact_path")
    if not artifact_path:
        return None, manifest
    path = Path(artifact_path)
    if not path.exists():
        return None, manifest
    try:
        return joblib.load(path), manifest
    except Exception:
        return None, manifest


def _heuristic_fallback(request: ExAnteDecisionRequest, warning: str = "heuristic_fallback") -> DecisionResult:
    try:
        from knowledge_base import HAND_KEY_BY_LABEL, HAND_RANKINGS
    except Exception:
        HAND_KEY_BY_LABEL = {}
        HAND_RANKINGS = {}

    hand = str(request.hand or "").replace(" ", "").upper()
    key = HAND_KEY_BY_LABEL.get(hand)
    tier = int(HAND_RANKINGS.get(key, {}).get("tier", 0)) if key else 0

    position = str(request.position or "").upper()
    stack_bb = float(request.stack_bb or 0.0)
    is_short = stack_bb <= 12

    if tier and tier <= 2:
        action = "RAISE"
        confidence = 0.66
    elif tier and tier <= 4:
        action = "CALL" if position in {"BTN", "CO", "SB"} else "FOLD"
        confidence = 0.56
    else:
        action = "FOLD"
        confidence = 0.72

    if is_short and tier and tier <= 4:
        action = "RAISE"
        confidence = max(confidence, 0.60)

    if request.is_3bet_spot and tier and tier > 4:
        action = "FOLD"

    return DecisionResult(
        action=action,
        confidence=round(confidence, 3),
        warning=warning,
        abstained=False,
        rationale=f"heuristic tier={tier} position={position} stack_bb={stack_bb:.1f}",
        source="heuristic",
        schema_version=EX_ANTE_SCHEMA_VERSION,
        model_version="heuristic",
        feature_version=EX_ANTE_SCHEMA_VERSION,
        metadata={"tier": tier, "position": position, "stack_bb": stack_bb},
    )


class DecisionService:
    def __init__(self, model: Any | None = None, manifest: dict[str, Any] | None = None) -> None:
        self.model = model
        self.manifest = manifest or {}
        self.contract = ExAnteFeatureContract()

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("model_version", "heuristic"))

    def predict(self, request: ExAnteDecisionRequest | dict[str, Any]) -> DecisionResult:
        if isinstance(request, dict):
            request = ExAnteDecisionRequest(**request)

        if not request.hand or not request.position or request.stack_bb <= 0 or request.num_players < 2:
            return DecisionResult(
                action="WARNING",
                confidence=0.0,
                warning="abstain_missing_required_context",
                abstained=True,
                rationale="missing_required_context",
                source="abstain",
                schema_version=EX_ANTE_SCHEMA_VERSION,
                model_version=self.model_version,
                feature_version=EX_ANTE_SCHEMA_VERSION,
            )

        try:
            feature_row = self.contract.build_row(request)
        except ValueError as exc:
            return DecisionResult(
                action="WARNING",
                confidence=0.0,
                warning=str(exc),
                abstained=True,
                rationale="feature_contract_rejected_input",
                source="contract",
                schema_version=EX_ANTE_SCHEMA_VERSION,
                model_version=self.model_version,
                feature_version=EX_ANTE_SCHEMA_VERSION,
            )

        if self.model is None:
            return _heuristic_fallback(request, warning="model_unavailable")

        frame = pd.DataFrame([feature_row])
        feature_columns = self.manifest.get("feature_columns")
        if feature_columns:
            for col in feature_columns:
                if col not in frame.columns:
                    frame[col] = 0
            frame = frame[list(feature_columns)]

        try:
            probabilities = self.model.predict_proba(frame)[0]
            classes = list(getattr(self.model, "classes_", self.manifest.get("classes", [])))
            prob_map = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
            action = max(prob_map, key=prob_map.get)
            confidence = float(max(probabilities))
        except Exception as exc:
            return DecisionResult(
                action="WARNING",
                confidence=0.0,
                warning=f"model_inference_error:{exc}",
                abstained=True,
                rationale="inference_error",
                source="error",
                schema_version=EX_ANTE_SCHEMA_VERSION,
                model_version=self.model_version,
                feature_version=EX_ANTE_SCHEMA_VERSION,
            )

        if confidence < CONFIDENCE_THRESHOLD:
            return DecisionResult(
                action="WARNING",
                confidence=round(confidence, 3),
                warning="low_confidence_abstain",
                abstained=True,
                rationale=f"low_confidence:{action}",
                source="model",
                schema_version=EX_ANTE_SCHEMA_VERSION,
                model_version=self.model_version,
                feature_version=EX_ANTE_SCHEMA_VERSION,
                probabilities=prob_map,
                metadata={"feature_row": feature_row},
            )

        return DecisionResult(
            action=action,
            confidence=round(confidence, 3),
            warning="",
            abstained=False,
            rationale=f"model_prediction:{action}",
            source="model",
            schema_version=EX_ANTE_SCHEMA_VERSION,
            model_version=self.model_version,
            feature_version=EX_ANTE_SCHEMA_VERSION,
            probabilities=prob_map,
            metadata={"feature_row": feature_row},
        )


_SERVICE: DecisionService | None = None


def get_decision_service() -> DecisionService:
    global _SERVICE
    if _SERVICE is None:
        model, manifest = _load_latest_model()
        _SERVICE = DecisionService(model=model, manifest=manifest)
    return _SERVICE


def predict_ex_ante(request: ExAnteDecisionRequest | dict[str, Any]) -> DecisionResult:
    return get_decision_service().predict(request)
