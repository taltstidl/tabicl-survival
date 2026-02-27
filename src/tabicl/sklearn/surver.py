from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival

from tabicl import TabICL, InferenceConfig
from tabicl.sklearn.preprocessing import TransformToNumerical


class TabICLSurver(SurvivalAnalysisMixin, BaseEstimator):
    def __init__(
            self,
            use_amp: bool = True,
            model_path: Optional[str | Path] = None,
            allow_auto_download: bool = True,
            checkpoint_version: str = "tabicl-survival-v1.ckpt",
            device: Optional[str | torch.device] = None,
            verbose: bool = False,
            inference_config: Optional[InferenceConfig | Dict] = None,
    ):
        self.use_amp = use_amp
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.verbose = verbose
        self.inference_config = inference_config

    def _load_model(self):
        repo_id = 'taltstidl/tabicl-survival'
        filename = self.checkpoint_version

        if self.model_path is None:
            # Scenario 1: the model path is not provided, so download from HF Hub based on the checkpoint version
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                    model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    )
            if model_path_:
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:
            # Scenario 2: the model path is provided
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                # Scenario 2a: the model path exists, load it directly
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                # Scenario 2b: the model path does not exist, download the checkpoint version to this path
                if self.allow_auto_download:
                    print(
                        f"Checkpoint not found at '{model_path_}'.\n"
                        f"Downloading '{filename}' from Hugging Face Hub ({repo_id}) to this location.\n"
                    )
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                        f"'{filename}' from Hugging Face Hub ({repo_id})."
                    )

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
        self.X_ = self.X_encoder_.fit_transform(X)
        time = (time - time.min()) / (time.max() - time.min())
        self.y_ = np.stack([event, time], axis=-1)

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = self.X_encoder_.transform(X)

        X = torch.from_numpy(np.concatenate([self.X_, X], axis=0)).float().to(self.device_)
        y = torch.from_numpy(self.y_).float().to(self.device_)

        with torch.no_grad():
            outputs = self.model_(X.unsqueeze(0), y.unsqueeze(0), inference_config=self.inference_config_)
            return outputs.squeeze(0).squeeze(-1).cpu().numpy()
