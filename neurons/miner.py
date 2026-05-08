"""Poker44 miner: trained ensemble (RF + HGBM) on sanitized chunk features, with heuristic fallback."""


import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.training import calibration as poker44_calibration  # noqa: F401 -- joblib pickle types
from poker44.training.features import FEATURE_VERSION, featurize_chunk
from poker44.training.risk_postprocess import temperature_scale_probability
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]


class Miner(BaseMinerNeuron):
    """
    Chunk-level bot-risk miner.

    Loads ``scripts/miner/training/artifacts/chunk_model.joblib`` (or ``POKER44_CHUNK_MODEL_PATH``)
    when ``feature_version`` matches ``poker44.training.features.FEATURE_VERSION``.
    The bundle ``classifier`` is usually a sklearn ``VotingClassifier`` (RF + HGBM), optionally
    wrapped as ``poker44.training.calibration.PlattCalibratedClassifier`` when trained with
    ``train_model.py --calibrate``.

    Inference tuning (optional env):

    - ``POKER44_RISK_TEMPERATURE`` — logit temperature on raw P(bot); ``1.0`` disables.
      Values ``>1`` soften scores toward ``0.5`` (often lowers human FPR on eval).
    - ``POKER44_BOT_THRESHOLD`` — threshold on ``risk_scores`` for ``synapse.predictions``
      only (validator scoring uses ``risk_scores``; default ``0.5``).
    """

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @staticmethod
    def _manifest_implementation_files(repo_root: Path) -> list[Path]:
        """
        Paths hashed into ``implementation_sha256`` for competition traceability.

        Covers inference: miner entrypoint plus feature extraction, calibration hook, score post-processing.
        """
        here = Path(__file__).resolve()
        extras = (
            repo_root / "poker44" / "training" / "features.py",
            repo_root / "poker44" / "training" / "calibration.py",
            repo_root / "poker44" / "training" / "risk_postprocess.py",
        )
        out = [here]
        for p in extras:
            if p.is_file():
                out.append(p.resolve())
        return out

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self._ml_bundle: Optional[dict[str, Any]] = None
        self._risk_temperature = Miner._env_float("POKER44_RISK_TEMPERATURE", 1.0)
        self._bot_threshold = Miner._env_float("POKER44_BOT_THRESHOLD", 0.5)
        self._bot_threshold = max(1e-6, min(1.0 - 1e-6, self._bot_threshold))

        default_manifest_path = (
            repo_root / "scripts" / "miner" / "training" / "artifacts" / "chunk_model.joblib"
        )
        model_path = Path(os.getenv("POKER44_CHUNK_MODEL_PATH", str(default_manifest_path)))
        if joblib is not None and model_path.is_file():
            try:
                bundle = joblib.load(model_path)
                if bundle.get("feature_version") == FEATURE_VERSION:
                    self._ml_bundle = bundle
                    bt.logging.info(
                        f"Loaded chunk classifier | path={model_path} "
                        f"feature_version={FEATURE_VERSION} "
                        f"train_samples={bundle.get('train_samples')} "
                        f"calibrated={bundle.get('calibrated', False)} "
                        f"classifier={type(bundle['classifier']).__name__} "
                        f"risk_temperature={self._risk_temperature} "
                        f"bot_threshold={self._bot_threshold}"
                    )
                else:
                    bt.logging.warning(
                        "Chunk model feature_version mismatch; using heuristic. "
                        f"bundle={bundle.get('feature_version')} code={FEATURE_VERSION}"
                    )
            except Exception as exc:  # pragma: no cover
                bt.logging.warning(f"Could not load chunk model from {model_path}: {exc}")

        heuristic_defaults = {
            "model_name": "poker44-reference-heuristic",
            "model_version": "1",
            "framework": "python-heuristic",
            "license": "MIT",
            "repo_url": "https://github.com/Poker44/Poker44-subnet",
            "notes": "Reference heuristic miner shipped with the Poker44 subnet.",
            "open_source": True,
            "inference_mode": "remote",
            "training_data_statement": (
                "Reference heuristic miner. No training step. Uses only runtime chunk features."
            ),
            "training_data_sources": ["none"],
            "private_data_attestation": (
                "This reference miner does not train on validator-only evaluation data."
            ),
        }
        ml_defaults = {
            "model_name": "p44-structured-action-chunk-detector",
            "model_version": "3",
            "framework": "scikit-learn VotingClassifier (RF+HGBM), optional Platt calibration",
            # Do not inherit the reference subnet URL/name pairing; declare your public fork via env
            # (``POKER44_MODEL_REPO_URL`` / ``POKER44_MODEL_REPO_COMMIT`` or ``run_miner.sh`` git defaults).
            "repo_url": "",
            "repo_commit": "",
            "notes": (
                "Trained via scripts/miner/training/train_model.py: synthetic chunks plus the "
                "public Poker44 training benchmark JSONL when present; --calibrate adds "
                "poker44.training.calibration.PlattCalibratedClassifier when supported."
            ),
            "training_data_statement": (
                "Synthetic behavioral regimes plus released Poker44 public benchmark chunks with "
                "ground-truth labels; released benchmark chunks are training-only and not reused in "
                "the live competitive field."
            ),
            "training_data_sources": [
                "synthetic_poker44_training",
                "https://api.poker44.net/api/v1/benchmark/releases",
                "scripts/miner/training/benchmark/benchmark_prepared.jsonl",
            ],
            "private_data_attestation": (
                "Does not use validator evaluation payloads for training."
            ),
        }
        defaults = {**heuristic_defaults, **(ml_defaults if self._ml_bundle else {})}

        bt.logging.info(
            "🤖 Poker44 Miner started | mode="
            + ("trained_ensemble" if self._ml_bundle else "heuristic_fallback")
        )
        impl_files = Miner._manifest_implementation_files(repo_root)
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=impl_files,
            defaults=defaults,
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """
        One bot-risk score per chunk in ``[0, 1]``.

        When a bundle is loaded (``self._ml_bundle``), uses ``bundle['classifier'].predict_proba``
        on ``featurize_chunk`` vectors — raw ensemble or Platt-calibrated artifact from training.
        Otherwise uses the reference heuristic.
        """
        chunks = synapse.chunks or []
        if self._ml_bundle is not None:
            scores = [self._trained_ensemble_risk(c) for c in chunks]
            mode = "trained_ensemble"
        else:
            scores = [self._heuristic_chunk_risk(c) for c in chunks]
            mode = "heuristic"
        synapse.risk_scores = scores
        thr = self._bot_threshold
        synapse.predictions = [bool(float(s) >= thr) for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner Predictions ({mode}): {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks ({mode}).")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    def _chunk_classifier(self) -> Any:
        """Sklearn estimator from the joblib bundle (``VotingClassifier`` or ``PlattCalibratedClassifier``)."""
        assert self._ml_bundle is not None
        return self._ml_bundle["classifier"]

    def _trained_ensemble_risk(self, chunk: list[dict]) -> float:
        """Joblib bundle → feature vector → P(bot) (including Platt calibration if saved)."""
        if not chunk:
            return 0.5
        assert self._ml_bundle is not None
        x = featurize_chunk(chunk)
        clf = self._chunk_classifier()
        raw = float(clf.predict_proba(x.reshape(1, -1))[0, 1])
        adjusted = temperature_scale_probability(raw, self._risk_temperature)
        return round(self._clamp01(adjusted), 6)

    def _heuristic_chunk_risk(self, chunk: list[dict]) -> float:
        """Reference per-hand heuristic, averaged over the chunk."""
        if not chunk:
            return 0.5
        hand_scores = [self._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)
        return round(self._clamp01(avg_score), 6)

    def score_chunk(self, chunk: list[dict]) -> float:
        """Same routing as ``forward``: ensemble if loaded, else heuristic (for tests/tools)."""
        if self._ml_bundle is not None:
            return self._trained_ensemble_risk(chunk)
        return self._heuristic_chunk_risk(chunk)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    miner = Miner()
    miner.run()
