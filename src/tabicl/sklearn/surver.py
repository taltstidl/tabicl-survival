from typing import Optional, Dict, List

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival

from tabicl import TabICL, InferenceConfig
from tabicl.sklearn.preprocessing import TransformToNumerical, EnsembleGenerator


class TabICLSurver(SurvivalAnalysisMixin, BaseEstimator):
    def __init__(
            self,
            n_estimators: int = 32,
            norm_methods: Optional[str | List[str]] = None,
            feat_shuffle_method: str = "latin",
            outlier_threshold: float = 4.0,
            use_amp: bool = True,
            batch_size: Optional[int] = 8,
            device: Optional[str | torch.device] = None,
            random_state: int | None = 42,
            verbose: bool = False,
            inference_config: Optional[InferenceConfig | Dict] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config

    def _load_model(self):
        model_path = '../sklearn/step-10000.ckpt'
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

        assert 'config' in checkpoint, 'The checkpoint doesn\'t contain the model configuration.'
        assert 'state_dict' in checkpoint, 'The checkpoint doesn\'t contain the model state.'

        self.model_ = TabICL(**checkpoint['config'])
        self.model_.load_state_dict(checkpoint['state_dict'])
        self.model_.eval()

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)
        event, time = check_array_survival(X, y)

        if self.device is None:
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device

        # Load the pre-trained TabICL model
        self._load_model()
        self.model_.to(self.device_)

        # Inference configuration
        init_config = {
            "COL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
            "ROW_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
            "ICL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
        }
        # If None, default settings in InferenceConfig
        if self.inference_config is None:
            self.inference_config_ = InferenceConfig()
            self.inference_config_.update_from_dict(init_config)
        # If dict, update default settings
        elif isinstance(self.inference_config, dict):
            self.inference_config_ = InferenceConfig()
            for key, value in self.inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            self.inference_config_.update_from_dict(init_config)
        # If InferenceConfig, use as is
        else:
            self.inference_config_ = self.inference_config

        # Transform input features
        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)

        # Fit ensemble generator to create multiple dataset views
        self.ensemble_generator_ = EnsembleGenerator(
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
            target_type="surv",
        )
        self.ensemble_generator_.fit(X, y)

        return self

    def _batch_forward(self, Xs, ys, shuffle_patterns=None):
        batch_size = self.batch_size or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)
        if shuffle_patterns is None:
            shuffle_patterns = [None] * n_batches
        else:
            shuffle_patterns = np.array_split(shuffle_patterns, n_batches)

        outputs = []
        for X_batch, y_batch, pattern_batch in zip(Xs, ys, shuffle_patterns):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)
            if pattern_batch is not None:
                pattern_batch = pattern_batch.tolist()

            with torch.no_grad():
                out = self.model_(
                    X_batch,
                    y_batch,
                    feature_shuffles=pattern_batch,
                    return_logits=True,
                    inference_config=self.inference_config_,
                )
            outputs.append(out.float().cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def predict(self, X):
        check_is_fitted(self)

        X = self.X_encoder_.transform(X)

        data = self.ensemble_generator_.transform(X)
        outputs = []
        for norm_method, (Xs, ys) in data.items():
            shuffle_patterns = self.ensemble_generator_.feature_shuffle_patterns_[norm_method]
            outputs.append(self._batch_forward(Xs, ys, shuffle_patterns))
        outputs = np.concatenate(outputs, axis=0)

        # Calculate mean predictions over all ensemble members
        outputs = np.mean(outputs, axis=0)

        # Opportunistically compute single risk or survival time
        if outputs.shape[1] == 1:
            outputs = outputs[:, 0]

        return outputs
