from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, PreTrainedModel

from megatron.hub.common.state import SafeTensorsStateSource, StateDict, StateSource


class PreTrainedBase(ABC):
    """
    Abstract base class for all pretrained models.

    This class provides a generic mechanism for managing model artifacts
    (e.g., config, tokenizer) with lazy loading. Subclasses that are
    decorated with `@dataclass` can define artifacts as fields with metadata
    specifying a loader method. The `model` itself is handled via a
    dedicated property that relies on the abstract `_load_model` method.

    Example:
        @dataclass
        class MyModel(PreTrainedBase):
            config: AutoConfig = field(
                init=False,
                metadata=artifact(loader="_load_config")
            )

            def _load_model(self) -> "PreTrainedModel":
                # Implementation for the loading logic
                ...
    """

    model_name_or_path: Union[str, Path]
    ARTIFACTS: ClassVar[List[str]] = []

    def __init__(self, **kwargs):
        self._state_dict_accessor: Optional[StateDict] = None
        self.init_kwargs = kwargs

    def save_artifacts(self, save_directory: Union[str, Path]):
        """
        Saves all loaded, generic artifacts that have a `save_pretrained` method
        to the specified directory. Note: This does not save the `model` attribute.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the config
        self.config.save_pretrained(save_path)

        # Iterate over registered artifacts to save them in a predictable order
        for name in self.ARTIFACTS:
            artifact = getattr(self, name)
            if not hasattr(artifact, "save_pretrained"):
                raise ValueError(f"Artifact {name} does not have a `save_pretrained` method.")
            artifact.save_pretrained(save_path)

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Subclasses must implement this to load the main model."""
        pass

    @abstractmethod
    def _load_config(self) -> AutoConfig:
        """Subclasses must implement this to load the model config."""
        pass

    @property
    def model(self) -> PreTrainedModel:
        """Lazily loads and returns the underlying model."""
        if not hasattr(self, "_model"):
            self._model = self._load_model()
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel):
        """Manually set the model."""
        self._model = value

    @property
    def config(self) -> AutoConfig:
        """Lazy load and return the model config."""
        if not hasattr(self, "_config"):
            self._config = self._load_config()
        return self._config

    @config.setter
    def config(self, value: AutoConfig):
        """Set the config manually."""
        self._config = value

    @property
    def state(self) -> StateDict:
        """
        Get the state dict accessor for pandas-like querying.

        This accessor can be backed by either a fully loaded model in memory
        or a ".safetensors" checkpoint on disk, enabling lazy loading of tensors.

        Examples:
            model.state()  # Get full state dict
            model.state["key"]  # Get single entry
            model.state[["key1", "key2"]]  # Get multiple entries
            model.state["*.weight"]  # Glob pattern
            model.state.regex(r".*\\.bias$")  # Regex pattern
        """
        if self._state_dict_accessor is None:
            source: Optional[Union[Dict[str, torch.Tensor], StateSource]] = None
            # Prioritize the loaded model's state_dict if available
            if hasattr(self, "_model") and self._model is not None:
                source = self.model.state_dict()
            elif hasattr(self, "model_name_or_path") and self.model_name_or_path:
                source = SafeTensorsStateSource(self.model_name_or_path)

            if source is None:
                raise ValueError(
                    "Cannot create StateDict accessor: model is not loaded and model_name_or_path is not set."
                )
            self._state_dict_accessor = StateDict(source)
        return self._state_dict_accessor
