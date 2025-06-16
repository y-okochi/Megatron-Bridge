from typing import Generic, TypeVar, overload, Iterable, Literal, Any, Type, Union
from pathlib import Path
from functools import partial
import dataclasses

import torch.distributed
import transformers
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig, MLATransformerConfig

from megatron.hub.bridge.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.hub.common.state import SafeTensorsStateSource
from megatron.hub.bridge import model_bridge
from megatron.hub.bridge.model_bridge import WeightDistributionMode
from megatron.hub.models.gpt_provider import GPTModelProvider

MegatronModelT = TypeVar("ModelT", bound=MegatronModule)
DataclassT = TypeVar("DataclassT")


class CausalLMBridge(Generic[MegatronModelT]):
    def __init__(self, hf_pretrained: PreTrainedCausalLM | PretrainedConfig):
        if not isinstance(hf_pretrained, (PreTrainedCausalLM, PretrainedConfig)):
            raise ValueError("hf_pretrained must be a PreTrainedCausalLM or PretrainedConfig instance")
        self.hf_pretrained: PreTrainedCausalLM | PretrainedConfig = hf_pretrained

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """
        List all model architectures currently supported by the bridge system.
        
        Returns:
            List of supported HuggingFace model architecture names
        """
        # Get all registered implementations from the dispatch system
        supported = []
        
        # Access the dispatch registry to find all registered types
        if hasattr(model_bridge.to_megatron, '_exact_types'):
            for arch_type in model_bridge.to_megatron._exact_types.keys():
                if hasattr(arch_type, '__name__'):
                    supported.append(arch_type.__name__)
        
        return sorted(supported)

    @classmethod
    def supports(cls, config: Any) -> bool:
        """
        Check if this bridge supports the given model configuration.
        
        A model is supported if it has at least one architecture ending with 'ForCausalLM'.
        
        Args:
            config: HuggingFace model config object
            
        Returns:
            True if this bridge can handle the model, False otherwise
        """
        architectures = getattr(config, 'architectures', [])
        if not architectures:
            return False
        
        return any(arch.endswith('ForCausalLM') for arch in architectures)
    
    @classmethod
    def from_config(cls, config: PretrainedConfig) -> "CausalLMBridge":
        cls._validate_config(config)
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> "CausalLMBridge":
        """
        Load a CausalLMBridge from a pretrained model.
        
        Args:
            path: HuggingFace model ID or path to model
            **kwargs: Additional arguments passed to HuggingFace from_pretrained
            
        Returns:
            CausalLMBridge instance
            
        Raises:
            ValueError: If the model architecture is not supported
        """
        # First load just the config to check architecture support
        config = AutoConfig.from_pretrained(path)
        cls._validate_config(config, path)
        
        return cls(PreTrainedCausalLM.from_pretrained(path, **kwargs))

    @overload
    def __call__(
        self,
        model: list[MegatronModelT],
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable[model_bridge.HFWeightTuple]: ...

    def __call__(
        self,
        model,
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable[model_bridge.HFWeightTuple]:
        return self.export_weights(model=model, order=order, cpu=cpu, show_progress=show_progress, mode=mode)
    
    def load_weights(self, model: list[MegatronModelT], hf_path: str | Path | None = None) -> None:
        if hf_path is None:
            if not isinstance(self.hf_pretrained, PreTrainedCausalLM):
                raise ValueError("hf_path is required when hf_pretrained is not a PreTrainedCausalLM instance")
            pre_trained = self.hf_pretrained
        else:
            pre_trained = PreTrainedCausalLM.from_pretrained(hf_path)
        self._model_bridge.load_weights(model, pre_trained)

        return model
    
    @overload
    def export_weights(
        self,
        model: list[MegatronModelT],
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable[model_bridge.HFWeightTuple]: ...

    def export_weights(
        self,
        model,
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable[model_bridge.HFWeightTuple]:
        query = (self._get_causal_lm_architecture(), self._get_model_instance(model))
        return model_bridge.bridge_state_to_hf(
            query, model, self.hf_pretrained, order=order, cpu=cpu, show_progress=show_progress, mode=mode
        )

    @overload
    def save_pretrained(
        self, model: list[MegatronModelT], path: str | Path
    ) -> None: ...

    def save_pretrained(self, model, path: str | Path, show_progress: bool = True) -> None:
        if torch.distributed.get_rank() == 0:
            self.hf_pretrained.save_artifacts(path)

        self.save_weights(model, path, show_progress)
        
    @overload
    def save_weights(
        self, model: list[MegatronModelT], path: str | Path, show_progress: bool = True
    ) -> None: ...
    
    def save_weights(self, model, path: str | Path, show_progress: bool = True) -> None:
        torch.distributed.barrier()
        query = (self._get_causal_lm_architecture(), self._get_model_instance(model))
        generator = model_bridge.bridge_state_to_hf(
            query, model, self.hf_pretrained, order="safetensors", cpu=True, show_progress=show_progress
        )

        # Check if the state source is SafeTensorsStateSource for streaming save.
        if (
            hasattr(self.hf_pretrained, "state")
            and hasattr(self.hf_pretrained.state, "source")
            and isinstance(self.hf_pretrained.state.source, SafeTensorsStateSource)
        ):
            self.hf_pretrained.state.source.save_generator(generator, path)
        else:
            raise ValueError(
                "The state source is not a SafeTensorsStateSource, cannot save in streaming mode."
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def push_to_hub(self, path: str | Path) -> None: ...

    def bridge_state_to_megatron(self) -> Iterable[model_bridge.MegatronWeightTuple]:
        return self._model_bridge.bridge_state_to_megatron(self.hf_pretrained)

    def to_megatron(self, load_weights: bool = True) -> GPTModelProvider:
        provider = self._model_bridge.provider_bridge(self.hf_pretrained)

        if load_weights:
            provider.model_transform = partial(self._model_bridge.load_state_from_hf, self.hf_pretrained)
        
        return provider
        
    @property
    def transformer_config(self) -> TransformerConfig:
        _model_provider = self.to_megatron(load_weights=False)
        return self._create_config_from_provider(_model_provider, TransformerConfig)
    
    @property
    def mla_transformer_config(self) -> MLATransformerConfig:
        _model_provider = self.to_megatron(load_weights=False)
        return self._create_config_from_provider(_model_provider, MLATransformerConfig)

    @property
    def _model_bridge(self) -> model_bridge.MegatronModelBridge:
        return model_bridge.get_model_bridge(self._get_causal_lm_architecture())

    def _get_causal_lm_architecture(self):
        """
        Get the CausalLM architecture class from the HuggingFace model.
        
        Returns:
            The transformers class for the CausalLM architecture
            
        Raises:
            ValueError: If no CausalLM architecture is found or if the class cannot be imported
        """
        if isinstance(self.hf_pretrained, PreTrainedCausalLM):
            architectures = getattr(self.hf_pretrained.config, 'architectures', [])
        else:
            architectures = getattr(self.hf_pretrained, 'architectures', [])
        
        if not architectures:
            raise ValueError(
                f"\n✗ No architectures found in model config\n\n"
                f"The model configuration does not specify any architectures.\n"
                f"This is required for determining the model type."
            )
        
        causal_lm_arch = None
        for architecture_name in architectures:
            if architecture_name.endswith("ForCausalLM"):
                causal_lm_arch = architecture_name
                break
        
        if not causal_lm_arch:
            raise ValueError(
                f"\n✗ No CausalLM architecture found\n\n"
                f"Model architectures: {architectures}\n\n"
                f"None of the architectures end with 'ForCausalLM'.\n"
                f"This bridge only supports causal language models.\n"
                f"For other model types, use a different bridge class."
            )
        
        try:
            return getattr(transformers, causal_lm_arch)
        except AttributeError:
            raise ValueError(
                f"\n✗ Architecture class '{causal_lm_arch}' not found in transformers\n\n"
                f"This could mean:\n"
                f"1. The model requires a newer version of transformers\n"
                f"2. The model uses a custom modeling file not in the standard library\n"
                f"3. There's a typo in the architecture name\n\n"
                f"Please verify your transformers installation and the model requirements."
            )
    
    @classmethod
    def _validate_config(cls, config: PretrainedConfig, path: str | None = None) -> None:
        # Check if this is a causal LM model
        if not cls.supports(config):
            architectures = getattr(config, 'architectures', [])
            raise ValueError(
                f"\n✗ Model architecture not supported by CausalLMBridge\n\n"
                f"Model: {path}\n"
                f"Architectures: {architectures}\n\n"
                f"CausalLMBridge only supports models with architectures ending in 'ForCausalLM'.\n"
                f"Found architectures that don't match this pattern.\n\n"
                f"If this is a different model type (e.g., Vision, Sequence-to-Sequence),\n"
                f"you may need to use a different bridge class."
            )
        
        # Check if we have an implementation for this specific architecture
        architecture = None
        for arch_name in config.architectures:
            if arch_name.endswith("ForCausalLM"):
                architecture = arch_name
                break
        
        if architecture:
            # Try to get the transformers class to check dispatch registration
            try:
                arch_class = getattr(transformers, architecture)
                # Test if we have a registered implementation
                # Check if this architecture is registered in the dispatch system
                has_implementation = False
                if hasattr(model_bridge.to_megatron, '_exact_types'):
                    has_implementation = arch_class in model_bridge.to_megatron._exact_types
                
                if not has_implementation:
                    # Get list of supported models
                    supported_models = cls.list_supported_models()
                    
                    raise ValueError(
                        f"\n✗ Model architecture '{architecture}' is not yet supported\n\n"
                        f"Model: {path}\n"
                        f"Architecture: {architecture}\n\n"
                        f"Currently supported architectures:\n" +
                        "\n".join(f"  • {model}" for model in supported_models) +
                        f"\n\nTo add support for {architecture}, you need to:\n"
                        f"1. Create a new bridge class that inherits from MegatronModelBridge\n"
                        f"2. Implement the required methods (provider_bridge, state_bridge)\n"
                        f"3. Register it with @MegatronModelBridge.impl decorator\n\n"
                        f"Example implementation:\n"
                        f"  from mhub.hub.bridge.model_bridge import MegatronModelBridge\n"
                        f"  from transformers import {architecture}\n"
                        f"  from megatron.core.models.gpt import GPTModel\n\n"
                        f"  @MegatronModelBridge.impl(source={architecture}, target=GPTModel)\n"
                        f"  class Megatron{architecture.replace('ForCausalLM', '')}Bridge(MegatronModelBridge):\n"
                        f"      def provider_bridge(self, hf_pretrained):\n"
                        f"          # Return a ModelProvider instance\n"
                        f"          ...\n\n"
                        f"      def state_bridge(self):\n"
                        f"          # Return a MegatronStateBridge with weight mappings\n"
                        f"          ...\n\n"
                        f"For reference implementations, see:\n"
                        f"  • src/megatron/hub/models/llama/llama_causal_bridge.py\n"
                        f"  • src/megatron/hub/models/qwen/qwen_2_causal_bridge.py"
                    ) from None
            except AttributeError:
                raise ValueError(
                    f"\n✗ Could not find architecture class '{architecture}' in transformers\n\n"
                    f"This might be because:\n"
                    f"1. The transformers library version is too old\n"
                    f"2. The model requires a custom modeling file\n"
                    f"3. The architecture name is incorrect\n\n"
                    f"Please check your transformers installation and model requirements."
                )
    
    def _get_model_instance(self, model: list[MegatronModelT]) -> MegatronModelT:
        model_instance = model[0]
        while hasattr(model_instance, 'module'):
            model_instance = model_instance.module
        return model_instance

    def _create_config_from_provider(
        self, source_obj: Any, target_dataclass: Type[DataclassT]
    ) -> DataclassT:
        kwargs = {}
        for field in dataclasses.fields(target_dataclass):
            if hasattr(source_obj, field.name):
                kwargs[field.name] = getattr(source_obj, field.name)
        return target_dataclass(**kwargs)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        lines_for_build = []

        # Format hf_pretrained
        hf_repr_actual_lines = repr(self.hf_pretrained).splitlines()
        if hf_repr_actual_lines:
            # First line of hf_pretrained part
            lines_for_build.append(f"  (hf_pretrained): {hf_repr_actual_lines[0]}")
            # Subsequent lines of hf_pretrained part, indented
            for line in hf_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (hf_pretrained): ")  # Fallback for empty repr

        # Format to_megatron dispatcher
        tm_repr_actual_lines = repr(model_bridge.to_megatron).splitlines()
        if tm_repr_actual_lines:
            # First line of to_megatron part
            lines_for_build.append(f"  (to_megatron): {tm_repr_actual_lines[0]}")
            # Subsequent lines of to_megatron part, indented
            for line in tm_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (to_megatron): ")  # Fallback for empty repr

        return f"{class_name}(\n" + "\n".join(lines_for_build) + "\n)"
