"""
Automatic bridge selection for Megatron-Hub models.

This module provides AutoBridge, which automatically selects the appropriate
bridge based on the model configuration without requiring users to know which
specific bridge to use.

The AutoBridge examines model architectures and selects the first compatible
bridge from a predefined list. This allows users to work with models without
needing to understand the underlying bridge implementation details.

Example:
    >>> from megatron.hub import AutoBridge
    >>> 
    >>> # Automatically loads with the correct bridge type
    >>> bridge = AutoBridge.from_pretrained("meta-llama/Llama-3-8B")
    >>> 
    >>> # Convert to Megatron format
    >>> provider = bridge.to_megatron()
    >>> model = provider(wrap_with_ddp=False)
"""
from typing import List, Type, Union, Any, Protocol
from pathlib import Path

from transformers import AutoConfig

from megatron.hub.bridge.causal_bridge import CausalLMBridge


_BRIDGES: List[Type["BridgeProtocol"]] = [
    CausalLMBridge,
]


class AutoBridge:
    """
    Automatically select and instantiate the appropriate bridge for a model.
    
    This class examines the model configuration and selects the first bridge
    that supports it. No dynamic discovery or decorators are used - all bridges
    must be explicitly imported and added to the _BRIDGES list.
    
    The AutoBridge provides a user-friendly interface similar to HuggingFace's
    Auto classes, allowing seamless model loading without needing to know the
    specific bridge implementation required for each model architecture.
    
    Example:
        >>> from megatron.hub import AutoBridge
        >>> # Load a Llama model without knowing it needs CausalLMBridge
        >>> bridge = AutoBridge.from_pretrained("meta-llama/Llama-3-8B")
        >>> # Automatically returns a CausalLMBridge instance
        
        >>> # Works with local paths too
        >>> bridge = AutoBridge.from_pretrained("/path/to/model")
        
        >>> # Check if a model is supported before loading
        >>> if AutoBridge.can_handle("microsoft/phi-2"):
        ...     bridge = AutoBridge.from_pretrained("microsoft/phi-2")
        
        >>> # Pass additional arguments to the underlying bridge
        >>> bridge = AutoBridge.from_pretrained(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     torch_dtype=torch.float16,
        ...     device_map="auto",
        ...     trust_remote_code=True
        ... )
    
    Note:
        Currently supports CausalLM models. Additional model types can be
        supported by implementing the appropriate bridge and adding it to
        the _BRIDGES list.
    """
    
    @classmethod
    def from_pretrained(cls, path: Union[str, Path], **kwargs) -> "BridgeProtocol":
        """
        Load a pretrained model, automatically selecting the appropriate bridge.
        
        This method:
        1. Loads only the model configuration (no weights)
        2. Iterates through registered bridges to find one that supports it
        3. Uses that bridge to load the full model
        
        Args:
            path: Path to model directory or HuggingFace model ID
            **kwargs: Additional arguments passed to the bridge's from_pretrained
                method (e.g., trust_remote_code, device_map, etc.)
        
        Returns:
            An instance of the appropriate bridge with the model loaded
            
        Raises:
            ValueError: If no registered bridge supports the model
        """
        # Load only the configuration - this is fast and doesn't load weights
        try:
            config = AutoConfig.from_pretrained(
                path, 
                trust_remote_code=kwargs.get('trust_remote_code', False)
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {path}. "
                f"Ensure the path is valid and contains a config.json file. "
                f"Error: {e}"
            )
        
        # Try each bridge in order
        for bridge_cls in _BRIDGES:
            if hasattr(bridge_cls, 'supports') and bridge_cls.supports(config):
                # Found a supporting bridge - use it to load the model
                try:
                    return bridge_cls.from_pretrained(path, **kwargs)
                except Exception as e:
                    # Log but continue - maybe another bridge will work
                    # In production, you might want to use proper logging here
                    print(
                        f"Warning: {bridge_cls.__name__} supports the config but "
                        f"failed to load: {e}"
                    )
                    continue
        
        # No bridge found
        architectures = getattr(config, 'architectures', ['unknown'])
        model_type = getattr(config, 'model_type', 'unknown')
        
        raise ValueError(
            f"No bridge found for model at {path}. "
            f"Model type: {model_type}, architectures: {architectures}. "
            f"Available bridges: {[b.__name__ for b in _BRIDGES]}. "
            f"Please use a specific bridge directly or implement a new bridge "
            f"for this model type."
        )
    
    @classmethod
    def get_supported_bridges(cls) -> List[str]:
        """
        Get list of all registered bridge class names.
        
        This method is useful for debugging and for understanding which model
        types are currently supported by the AutoBridge system.
        
        Returns:
            List[str]: Bridge class names in priority order. The order matters
                as bridges are tried sequentially during model loading.
        
        Example:
            >>> bridges = AutoBridge.get_supported_bridges()
            >>> print(f"Supported bridges: {bridges}")
            ['CausalLMBridge']
        """
        return [bridge.__name__ for bridge in _BRIDGES]
    
    @classmethod
    def can_handle(cls, path: Union[str, Path], trust_remote_code: bool = False) -> bool:
        """
        Check if any registered bridge can handle the model at the given path.
        
        This method allows you to verify model compatibility before attempting
        to load it, which can be useful for validation or UI feedback.
        
        Args:
            path: Path to model directory or HuggingFace model ID
                Examples: "meta-llama/Llama-3-8B", "/models/my_model"
            trust_remote_code: Whether to trust remote code when loading config.
                Set to True for models that use custom modeling code.
            
        Returns:
            bool: True if at least one bridge supports the model, False otherwise
        
        Example:
            >>> # Check if a model is supported
            >>> if AutoBridge.can_handle("meta-llama/Llama-3-8B"):
            ...     print("Model is supported!")
            ... else:
            ...     print("Model requires a custom bridge implementation")
        """
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
            return any(
                hasattr(bridge_cls, 'supports') and bridge_cls.supports(config)
                for bridge_cls in _BRIDGES
            )
        except Exception:
            return False


class BridgeProtocol(Protocol):
    """
    Protocol defining the interface for model bridges.
    
    All bridges that want to participate in automatic selection must implement
    these methods. This is a typing-only protocol with no runtime checks.
    """
    
    @classmethod
    def supports(cls, config: Any) -> bool:
        """
        Check if this bridge supports the given model configuration.
        
        Args:
            config: HuggingFace model config object
            
        Returns:
            True if this bridge can handle the model, False otherwise
        """
        ...
    
    @classmethod
    def from_pretrained(cls, path: Union[str, Path], **kwargs) -> "BridgeProtocol":
        """
        Load a pretrained model using this bridge.
        
        Args:
            path: Path to the model (local directory or HuggingFace model ID)
            **kwargs: Additional arguments passed to the underlying loader
            
        Returns:
            Instance of the bridge with loaded model
        """
        ...


