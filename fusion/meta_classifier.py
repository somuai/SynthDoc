"""Meta-classifier fusion engine for SynthDoc.

Combines predictions from spatial, frequency, and semantic streams
using a calibrated XGBoost + LightGBM ensemble to produce a final
fraud probability and risk tier classification.
"""

import numpy as np
import joblib


# Document type numeric mapping
DOC_TYPE_MAP = {
    'PAN_CARD': 0,
    'AADHAAR': 1,
    'PASSPORT': 2,
    'VOTER_ID': 3,
    'DRIVING_LICENSE': 4,
    'UPI_QR': 5,
}

# Risk tier thresholds
RISK_THRESHOLDS = {
    'LOW': (0.0, 0.25),
    'MEDIUM': (0.25, 0.50),
    'HIGH': (0.50, 0.75),
    'CRITICAL': (0.75, 1.01),
}

# Feature order for the meta-classifier input vector
FEATURE_ORDER = [
    'spatial_score', 'frequency_score', 'semantic_score',
    'spatial_conf', 'frequency_conf', 'semantic_conf',
    'doc_type', 'resolution_norm', 'file_size_norm',
]


def _classify_risk_tier(probability: float) -> str:
    """Map a fraud probability to a risk tier string."""
    for tier, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return tier
    return 'CRITICAL'


class SynthDocMetaClassifier:
    """Ensemble meta-classifier combining XGBoost and LightGBM
    with Isotonic Regression calibration."""

    def __init__(self):
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self._is_trained = False

    def _features_to_array(self, features: dict) -> np.ndarray:
        """Convert a feature dictionary to a numpy array in the correct order."""
        return np.array([[features.get(f, 0.0) for f in FEATURE_ORDER]])

    def train(self, X: np.ndarray, y: np.ndarray, X_cal: np.ndarray = None, y_cal: np.ndarray = None):
        """Train both ensemble models and the calibrator.

        Args:
            X: Training feature matrix (N, 9).
            y: Binary labels (0=genuine, 1=synthetic).
            X_cal: Calibration set features (optional, uses X if None).
            y_cal: Calibration set labels.
        """
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)

        # Calibration
        if X_cal is None:
            X_cal, y_cal = X, y

        xgb_proba = self.xgb_model.predict_proba(X_cal)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X_cal)[:, 1]
        avg_proba = (xgb_proba + lgb_proba) / 2.0

        self.calibrator.fit(avg_proba, y_cal)
        self._is_trained = True

    def predict(self, features: dict) -> dict:
        """Predict fraud probability and risk tier from stream features.

        Args:
            features: Dictionary with keys matching FEATURE_ORDER.

        Returns:
            dict with 'fraud_probability' and 'risk_tier'.
        """
        X = self._features_to_array(features)

        if self._is_trained:
            xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
            lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
            avg_proba = (xgb_proba + lgb_proba) / 2.0
            calibrated = float(self.calibrator.predict(avg_proba)[0])
        else:
            # Fallback: weighted average of stream scores
            calibrated = (
                features.get('spatial_score', 0.5) * 0.4 +
                features.get('frequency_score', 0.5) * 0.35 +
                features.get('semantic_score', 0.5) * 0.25
            )

        calibrated = float(np.clip(calibrated, 0.0, 1.0))
        risk_tier = _classify_risk_tier(calibrated)

        return {
            'fraud_probability': calibrated,
            'risk_tier': risk_tier,
        }

    def save(self, path: str) -> None:
        """Save the trained ensemble to disk."""
        state = {
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'calibrator': self.calibrator,
            'is_trained': self._is_trained,
        }
        joblib.dump(state, path)

    def load(self, path: str) -> None:
        """Load a trained ensemble from disk."""
        state = joblib.load(path)
        self.xgb_model = state['xgb_model']
        self.lgb_model = state['lgb_model']
        self.calibrator = state['calibrator']
        self._is_trained = state['is_trained']
