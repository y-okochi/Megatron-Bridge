from typing import Optional, Union, TypeVar, Generic, Dict, Any, List
from pathlib import Path
import sys

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    ProcessorMixin,
)
from transformers.generation.utils import GenerateOutput

from mhub.hub._lib.hf.base import PreTrainedBase

# Python 3.12+ supports PEP 692 (TypedDict Unpack)
if sys.version_info >= (3, 12):
    from typing import TypedDict, Unpack
else:
    from typing_extensions import TypedDict, Unpack


# Type variable for generic model type
VLMType = TypeVar("VLMType", bound=PreTrainedModel)


class PreTrainedVLM(PreTrainedBase, Generic[VLMType]):
    """
    A generic class for Pretrained Vision-Language Models with lazy loading.

    Allows type-safe access to specific VLM implementations like LlavaForConditionalGeneration.

    Examples:
        Basic usage with image and text:
        >>> from mbridge.pretrained import PreTrainedVLM
        >>> from PIL import Image
        >>>
        >>> # Create instance - no model loading happens yet
        >>> vlm = PreTrainedVLM.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>>
        >>> # Load an image
        >>> image = Image.open("cat.jpg")
        >>>
        >>> # Process image and text together - processor and model load here
        >>> inputs = vlm.process_images_and_text(
        ...     images=image,
        ...     text="What do you see in this image?"
        ... )
        >>>
        >>> # Generate response
        >>> outputs = vlm.generate(**inputs, max_new_tokens=100)
        >>> print(vlm.decode(outputs[0], skip_special_tokens=True))

        Batch processing with multiple images:
        >>> # Process multiple images with questions
        >>> images = [Image.open(f"image_{i}.jpg") for i in range(3)]
        >>> questions = [
        ...     "What is the main object in this image?",
        ...     "Describe the scene",
        ...     "What colors do you see?"
        ... ]
        >>>
        >>> # Process batch
        >>> inputs = vlm.process_images_and_text(
        ...     images=images,
        ...     text=questions,
        ...     padding=True
        ... )
        >>>
        >>> # Generate responses
        >>> outputs = vlm.generate(**inputs, max_new_tokens=50)
        >>> for i, output in enumerate(outputs):
        ...     print(f"Image {i+1}: {vlm.decode(output, skip_special_tokens=True)}")

        Using specific VLM types with type hints:
        >>> from transformers import LlavaForConditionalGeneration
        >>> from mbridge.pretrained import PreTrainedVLM
        >>>
        >>> # Type-safe access to Llava-specific features
        >>> llava: PreTrainedVLM[LlavaForConditionalGeneration] = PreTrainedVLM.from_pretrained(
        ...     "llava-hf/llava-1.5-7b-hf",
        ...     torch_dtype=torch.float16,
        ...     device="cuda"
        ... )
        >>>
        >>> # Access model-specific attributes
        >>> vision_tower = llava.model.vision_tower  # Type-safe access

        Text-only generation (for multimodal models that support it):
        >>> # Some VLMs can also work with text-only inputs
        >>> text_inputs = vlm.encode_text("Explain what a neural network is.")
        >>> outputs = vlm.generate(**text_inputs, max_length=100)
        >>> print(vlm.decode(outputs[0], skip_special_tokens=True))

        Custom preprocessing and generation:
        >>> # Load with custom settings
        >>> vlm = PreTrainedVLM.from_pretrained(
        ...     "Qwen/Qwen-VL-Chat",
        ...     trust_remote_code=True,
        ...     device_map="auto",
        ...     load_in_4bit=True
        ... )
        >>>
        >>> # Custom generation config
        >>> from transformers import GenerationConfig
        >>> vlm.generation_config = GenerationConfig(
        ...     max_new_tokens=200,
        ...     temperature=0.8,
        ...     top_p=0.95,
        ...     do_sample=True
        ... )
        >>>
        >>> # Process with custom parameters
        >>> inputs = vlm.process_images_and_text(
        ...     images=image,
        ...     text="<image>\\nDescribe this image in detail.",
        ...     max_length=512
        ... )

        Manual component setup:
        >>> # Create empty instance
        >>> vlm = PreTrainedVLM()
        >>>
        >>> # Load components separately
        >>> from transformers import AutoProcessor, AutoModel
        >>> vlm.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")
        >>> vlm.model = AutoModel.from_pretrained("microsoft/Florence-2-base")
        >>>
        >>> # Use for various vision tasks
        >>> task_prompt = "<OD>"  # Object detection task
        >>> inputs = vlm.process_images_and_text(images=image, text=task_prompt)
        >>> outputs = vlm.generate(**inputs)

        Conversational VLM usage:
        >>> # Multi-turn conversation with images
        >>> conversation = []
        >>>
        >>> # First turn
        >>> image1 = Image.open("chart.png")
        >>> inputs = vlm.process_images_and_text(
        ...     images=image1,
        ...     text="What type of chart is this?"
        ... )
        >>> response = vlm.generate(**inputs)
        >>> conversation.append(("user", "What type of chart is this?"))
        >>> conversation.append(("assistant", vlm.decode(response[0])))
        >>>
        >>> # Follow-up question
        >>> follow_up = "What is the highest value shown?"
        >>> # Format conversation history + new question
        >>> full_prompt = format_conversation(conversation) + f"\\nUser: {follow_up}"
        >>> inputs = vlm.process_images_and_text(images=image1, text=full_prompt)
        >>> response = vlm.generate(**inputs)
    """

    ARTIFACTS = ["processor", "tokenizer", "image_processor", "generation_config"]

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize a Pretrained VLM with lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on (e.g., 'cuda', 'cpu')
            torch_dtype: Data type to load model in (e.g., torch.float16)
            trust_remote_code: Whether to trust remote code when loading
        """
        self._model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        super().__init__()

    def _load_model(self) -> VLMType:
        """Lazy load and return the model."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            **self.init_kwargs,
        }

        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        # Use provided config if already loaded
        if "config" in self.__dict__:
            model_kwargs["config"] = self.config

        # Try AutoModel first for VLMs
        model = AutoModel.from_pretrained(self.model_name_or_path, **model_kwargs)

        # Move to device
        model = model.to(self.device)

        # Set generation config if available
        if "generation_config" in self.__dict__ and hasattr(
            model, "generation_config"
        ):
            model.generation_config = self.generation_config
        return model

    def _load_config(self) -> AutoConfig:
        """Lazy load and return the model config."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load config")

        return AutoConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **self.init_kwargs,
        )

    def _load_processor(self) -> ProcessorMixin:
        """Lazy load and return the processor."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load processor")

        try:
            return AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **self.init_kwargs,
            )
        except Exception:
            # Some VLMs might not have a processor, fall back to manual loading
            raise ValueError(
                f"Could not load processor for {self.model_name_or_path}. "
                "This model might require manual processor setup."
            )

    def _load_tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """
        Lazy load and return the tokenizer.
        For VLMs, the tokenizer might be included in the processor.
        """
        if "processor" in self.__dict__ and hasattr(self.processor, "tokenizer"):
            return self.processor.tokenizer

        # Try to load tokenizer separately
        if self.model_name_or_path is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )

                # Set padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
            except Exception:
                # Some VLMs include tokenizer only in processor
                pass
        return None

    def _load_image_processor(self) -> Optional[Any]:
        """
        Lazy load and return the image processor.
        For VLMs, the image processor might be included in the processor.
        """
        if "processor" in self.__dict__ and hasattr(
            self.processor, "image_processor"
        ):
            return self.processor.image_processor

        # Try to load image processor separately
        if self.model_name_or_path is not None:
            try:
                return AutoImageProcessor.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )
            except Exception:
                # Some VLMs include image processor only in processor
                pass
        return None

    def _load_generation_config(self) -> Optional[GenerationConfig]:
        """Lazy load and return the generation config."""
        if self.model_name_or_path is not None:
            try:
                return GenerationConfig.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )
            except Exception:
                # Not all models have generation configs
                pass
        return None

    @property
    def model_name_or_path(self) -> Optional[Union[str, Path]]:
        """Return the model name or path."""
        return self._model_name_or_path
    
    @property
    def processor(self) -> ProcessorMixin:
        """Lazy load and return the processor."""
        if self._processor is None:
            self._processor = self._load_processor()
        return self._processor
    
    @processor.setter
    def processor(self, value: ProcessorMixin):
        """Set the processor manually."""
        self._processor = value

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """Lazy load and return the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizer):
        """Set the tokenizer manually."""
        self._tokenizer = value
    
    @property
    def image_processor(self) -> Optional[Any]:
        """Lazy load and return the image processor."""
        if self._image_processor is None:
            self._image_processor = self._load_image_processor()
        return self._image_processor
    
    @image_processor.setter
    def image_processor(self, value: Any): 
        """Set the image processor manually."""
        self._image_processor = value
    
    @property
    def generation_config(self) -> Optional[GenerationConfig]:
        """Lazy load and return the generation config."""
        if self._generation_config is None:
            self._generation_config = self._load_generation_config()
        return self._generation_config
    
    @generation_config.setter
    def generation_config(self, value: GenerationConfig):
        """Set the generation config manually."""
        self._generation_config = value
        # Update model's generation config if model is loaded
        if self.model is not None and hasattr(self.model, "generation_config"):
            self.model.generation_config = value

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "PreTrainedVLM[VLMType]":
        """
        Create a PreTrainedVLM instance for lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on
            torch_dtype: Data type to load model in
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for from_pretrained methods

        Returns:
            PreTrainedVLM instance configured for lazy loading
        """
        return cls(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def process_images_and_text(
        self,
        images: Optional[Union[Any, List[Any]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Process images and text using the processor.

        Args:
            images: Single image or list of images (PIL, numpy, torch, etc.)
            text: Single text or list of texts
            **kwargs: Additional arguments for processor

        Returns:
            Dictionary with processed inputs ready for the model
        """
        if images is None and text is None:
            raise ValueError("At least one of images or text must be provided")

        processor_kwargs = {"return_tensors": "pt", **kwargs}

        # Handle different input combinations
        if images is not None and text is not None:
            inputs = self.processor(images=images, text=text, **processor_kwargs)
        elif images is not None:
            inputs = self.processor(images=images, **processor_kwargs)
        else:  # text only
            inputs = self.processor(text=text, **processor_kwargs)

        # Move to device
        if isinstance(inputs, dict):
            inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

        return inputs

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack["VLMGenerateKwargs"],
    ) -> Union[torch.LongTensor, GenerateOutput]:
        """
        Generate text from vision-language model inputs.

        This method forwards all arguments to the model's generate method,
        supporting multimodal generation with both text and image inputs.

        Common parameters include:
            inputs (torch.LongTensor, optional): Input token IDs for text-only generation.
            pixel_values (torch.FloatTensor, optional): Preprocessed image tensors
                for vision inputs. Shape: (batch_size, channels, height, width).
            input_ids (torch.LongTensor, optional): Text token IDs when used with
                pixel_values for multimodal generation.
            attention_mask (torch.Tensor, optional): Attention mask for input tokens.
            max_length (int, optional): Maximum length of generated sequence.
                Defaults to model's max_length configuration.
            min_length (int, optional): Minimum length of generated sequence.
            max_new_tokens (int, optional): Maximum number of tokens to generate,
                ignoring the number of tokens in the prompt.
            do_sample (bool, optional): Whether to use sampling. Defaults to False
                (greedy decoding).
            temperature (float, optional): Temperature for sampling. Higher values
                produce more random outputs. Typical range: 0.1-2.0.
            top_p (float, optional): Nucleus sampling threshold. Only tokens with
                cumulative probability up to top_p are considered. Range: 0.0-1.0.
            top_k (int, optional): Only consider the top k tokens for sampling.
            num_beams (int, optional): Number of beams for beam search. 1 means
                no beam search.
            repetition_penalty (float, optional): Penalty for repeating tokens.
                Values > 1.0 discourage repetition.
            pad_token_id (int, optional): ID of padding token.
            eos_token_id (int or List[int], optional): ID(s) of end-of-sequence token(s).
            use_cache (bool, optional): Whether to use past key values to speed up
                generation. Defaults to True.

        Returns:
            torch.LongTensor or transformers.generation.utils.GenerateOutput:
                Generated token IDs. If return_dict_in_generate=True, returns a
                GenerateOutput object containing generated sequences and additional
                information like scores.

        Examples:
            >>> # Text-only generation
            >>> model = PreTrainedVLM.from_pretrained("llava-hf/llava-1.5-7b-hf")
            >>> text_inputs = model.encode_text("Describe the weather:")
            >>> outputs = model.generate(**text_inputs, max_length=50)
            >>> print(model.decode(outputs[0]))

            >>> # Image + text generation
            >>> from PIL import Image
            >>> image = Image.open("weather.jpg")
            >>> inputs = model.process_images_and_text(
            ...     images=image,
            ...     text="What's the weather like in this image?"
            ... )
            >>> outputs = model.generate(**inputs, max_new_tokens=100)
            >>> response = model.decode(outputs[0])

            >>> # Batch generation with multiple images
            >>> images = [Image.open(f"img{i}.jpg") for i in range(3)]
            >>> texts = ["Describe this image:" for _ in range(3)]
            >>> inputs = model.process_images_and_text(images=images, text=texts)
            >>> outputs = model.generate(**inputs, max_length=100, num_beams=3)

        Note:
            - For vision-language models, the input format depends on the specific
              model architecture. Some models interleave image and text tokens,
              while others process them separately.
            - Use process_images_and_text() to properly format multimodal inputs.
            - For detailed documentation of all parameters, see the transformers
              library documentation for generation methods.
        """
        model = self.model
        # Sync generation config if it has been set on the wrapper
        if "generation_config" in self.__dict__ and hasattr(model, "generation_config"):
            model.generation_config = self.generation_config
        return model.generate(input_ids, **kwargs)

    def __call__(self, *args, **kwargs):
        """Forward call to model."""
        return self.model(*args, **kwargs)

    def encode_text(
        self, text: Union[str, List[str]], **kwargs: Unpack["VLMEncodeKwargs"]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using the tokenizer or processor for text-only inputs.

        This method handles text encoding for vision-language models, which may use
        either a standalone tokenizer or a processor that handles both modalities.
        The output is automatically moved to the same device as the model.

        Args:
            text (str or List[str]): Input text to encode. Can be a single string
                or a list of strings for batch encoding. For conversation-style
                models, can include special tokens or formatting.
            **kwargs: Additional arguments for encoding. Common options:
                padding (bool or str, optional): Padding strategy.
                    - True or 'longest': Pad to longest sequence in batch
                    - 'max_length': Pad to max_length
                    - False or 'do_not_pad': No padding (default)
                truncation (bool or str, optional): Truncation strategy.
                    - True or 'longest_first': Truncate to max_length
                    - False: No truncation
                max_length (int, optional): Maximum length of returned sequences.
                    Defaults to model's max_length.
                add_special_tokens (bool, optional): Whether to add special tokens
                    (e.g., <s>, </s>). Defaults to True.
                return_attention_mask (bool, optional): Whether to return attention
                    mask. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model inputs:
                - input_ids: Token IDs tensor of shape (batch_size, sequence_length)
                - attention_mask: Attention mask tensor of same shape
                - Additional keys may include position_ids, token_type_ids, etc.
                  depending on the model architecture.

        Examples:
            >>> model = PreTrainedVLM.from_pretrained("llava-hf/llava-1.5-7b-hf")
            >>> # Simple text encoding
            >>> text_inputs = model.encode_text("What is in this image?")
            >>> print(text_inputs.keys())  # dict_keys(['input_ids', 'attention_mask'])

            >>> # Batch encoding with padding
            >>> texts = ["Describe this:", "What do you see in the image?"]
            >>> text_inputs = model.encode_text(texts, padding=True)
            >>> print(text_inputs["input_ids"].shape)  # torch.Size([2, max_length])

            >>> # Encoding with conversation template (model-specific)
            >>> prompt = "USER: Analyze this image.\nASSISTANT:"
            >>> text_inputs = model.encode_text(prompt, add_special_tokens=True)

        Note:
            - This method is for text-only inputs. For multimodal inputs (text + images),
              use process_images_and_text() instead.
            - Some VLMs require specific text formatting or templates. Refer to the
              model's documentation for the expected input format.
            - If the model uses a processor instead of a tokenizer, this method
              will internally call process_images_and_text() with text-only inputs.
        """
        if self.tokenizer is not None:
            # Only set return_tensors default if not provided
            if "return_tensors" not in kwargs:
                kwargs["return_tensors"] = "pt"

            return self.tokenizer(text, **kwargs).to(self.device)
        else:
            # Use processor for text-only encoding
            return self.process_images_and_text(text=text, **kwargs)

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor],
        **kwargs: Unpack["VLMDecodeKwargs"],
    ) -> str:
        """
        Decode token IDs back into text using the model's tokenizer.

        This method converts token IDs (from model output or generation) back into
        human-readable text. It properly handles special tokens and formatting
        specific to vision-language models.

        Args:
            token_ids (int, List[int], or torch.Tensor): Token IDs to decode.
                Can be:
                - Single token ID (int)
                - List of token IDs
                - 1D tensor of token IDs
                - 2D tensor (will decode the first sequence)
            **kwargs: Additional arguments passed to the tokenizer's decode method:
                skip_special_tokens (bool, optional): Whether to remove special
                    tokens (e.g., <pad>, <s>, </s>, <image>) from output.
                    Defaults to True.
                clean_up_tokenization_spaces (bool, optional): Whether to clean up
                    tokenization artifacts (extra spaces, etc.). Defaults to True.

        Returns:
            str: Decoded text string with special tokens and formatting handled
                according to the model's conventions.

        Raises:
            ValueError: If no tokenizer is available for decoding (some VLMs only
                have processors without separate tokenizer access).

        Examples:
            >>> model = PreTrainedVLM.from_pretrained("llava-hf/llava-1.5-7b-hf")
            >>> # Decode generated output
            >>> inputs = model.process_images_and_text(image, "Describe this:")
            >>> output_ids = model.generate(**inputs, max_new_tokens=50)
            >>> text = model.decode(output_ids[0])
            >>> print(text)  # "Describe this: The image shows a sunny beach..."

            >>> # Decode with special tokens visible
            >>> text_with_special = model.decode(output_ids[0], skip_special_tokens=False)
            >>> print(text_with_special)  # "<s> Describe this: <image> The image shows..."

            >>> # Batch decoding (iterate over sequences)
            >>> for i, output in enumerate(output_ids):
            ...     decoded = model.decode(output)
            ...     print(f"Response {i}: {decoded}")

        Note:
            - Vision-language models may include special tokens for image placeholders
              (e.g., <image>, <img>, <IMAGE>) which are typically removed during decoding.
            - If decoding a full conversation, the output will include the entire
              sequence including the input prompt unless you slice the output_ids
              to only include newly generated tokens.
            - For batch decoding of multiple sequences, use tokenizer.batch_decode()
              directly or iterate over the sequences.
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer available for decoding")

        return tokenizer.decode(token_ids, **kwargs)

    def save_artifacts(self, save_directory: Union[str, Path]):
        """Saves all loaded artifacts, handling de-duplication for processor components."""
        artifacts_to_save = {}
        # Collect all loaded artifacts first
        for name in self._cls_artifacts.keys():
            if name in self.__dict__:
                artifacts_to_save[name] = self.__dict__[name]

        # If processor is loaded, it's the source of truth for its components.
        if "processor" in artifacts_to_save:
            processor = artifacts_to_save["processor"]
            # If tokenizer is also loaded and is the same object as the one in processor, remove it from our list.
            if "tokenizer" in artifacts_to_save and artifacts_to_save["tokenizer"] is getattr(processor, 'tokenizer', None):
                del artifacts_to_save["tokenizer"]
            # Same for image processor.
            if "image_processor" in artifacts_to_save and artifacts_to_save["image_processor"] is getattr(processor, 'image_processor', None):
                del artifacts_to_save["image_processor"]

        # Now save the de-duplicated artifacts
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        for name, artifact_instance in artifacts_to_save.items():
            if artifact_instance is not None and hasattr(artifact_instance, "save_pretrained"):
                artifact_instance.save_pretrained(save_path)
        
        # Manually save the model
        if self._model is not None:
            self.model.save_pretrained(save_path)

    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = device
        if self._model is not None:
            self.model = self.model.to(device)
        return self

    def half(self):
        """Convert model to half precision (float16)."""
        if self._model is not None:
            self.model = self.model.half()
        return self

    def float(self):
        """Convert model to full precision (float32)."""
        if self._model is not None:
            self.model = self.model.float()
        return self

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Get model's dtype if loaded."""
        if self._model is not None:
            try:
                return next(self.model.parameters()).dtype
            except StopIteration:
                return None
        return None

    @property
    def num_parameters(self) -> Optional[int]:
        """Get total number of parameters if model is loaded."""
        if self._model is not None:
            return sum(p.numel() for p in self.model.parameters())
        return None

    def __repr__(self) -> str:
        """Return a string representation of the PreTrainedVLM instance."""
        try:
            # Access config to trigger lazy loading for a richer repr
            _ = self.config
        except Exception:
            # If loading fails, repr shouldn't crash.
            pass

        lines = [f"{self.__class__.__name__}("]
        # Generic artifacts
        for name, _ in sorted(self.get_artifacts().items()):
            is_loaded = name in self.__dict__
            artifact_instance = self.__dict__.get(name)

            type_name = "N/A"
            details = "not loaded"
            if is_loaded and artifact_instance is not None:
                type_name = artifact_instance.__class__.__name__
                details = "loaded"
                if name == "tokenizer":
                    vocab = getattr(artifact_instance, "vocab_size", "N/A")
                    details = f"vocab_size={vocab}, loaded"
                elif name == "processor":
                    details = "loaded"
            lines.append(f"  ({name}): {type_name} [{details}]")
        
        # Manually add model repr
        model_repr_content: str
        if self._model is not None:
            model_class_name = self.model.__class__.__name__
            config = self.config
            details_list = []
            if hasattr(config, "vision_config"):
                details_list.append(f"vision_model={getattr(config.vision_config, 'model_type', 'N/A')}")
            if hasattr(config, "text_config"):
                details_list.append(f"text_model={getattr(config.text_config, 'model_type', 'N/A')}")
            details_str = ", ".join(details_list)
            model_repr_content = (
                f"{model_class_name} ({details_str}) [loaded]"
            )
        elif "config" in self.__dict__:
            config = self.config
            model_class_name_from_config = "VLM"
            if hasattr(config, "architectures") and config.architectures:
                model_class_name_from_config = config.architectures[0]
            model_repr_content = f"{model_class_name_from_config} [not loaded]"
        else:
            model_repr_content = "AutoModel [not loaded]"
        lines.append(f"  (model): {model_repr_content}")

        lines.sort()

        params_str = (
            f"{self.num_parameters:,}" if self.num_parameters is not None else "N/A"
        )
        dtype_str = (
            str(self.dtype).replace("torch.", "") if self.dtype is not None else "N/A"
        )
        lines.extend(
            [
                f"  (parameters): {params_str}",
                f"  (device): {str(self.device)}",
                f"  (dtype): {dtype_str}",
                ")",
            ]
        )
        return "\n".join(lines)


# TypedDict definitions for VLM method parameters
class VLMGenerateKwargs(TypedDict, total=False):
    """TypedDict for VLM generate method parameters."""

    pixel_values: Optional[torch.FloatTensor]
    attention_mask: Optional[torch.Tensor]
    max_length: Optional[int]
    max_new_tokens: Optional[int]
    min_length: Optional[int]
    do_sample: Optional[bool]
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    repetition_penalty: Optional[float]
    pad_token_id: Optional[int]
    eos_token_id: Optional[Union[int, List[int]]]
    bos_token_id: Optional[int]
    num_beams: Optional[int]
    num_return_sequences: Optional[int]
    early_stopping: Optional[bool]
    use_cache: Optional[bool]
    return_dict_in_generate: Optional[bool]
    output_scores: Optional[bool]
    output_attentions: Optional[bool]


class VLMEncodeKwargs(TypedDict, total=False):
    """TypedDict for VLM encode_text method parameters."""

    padding: Union[bool, str]
    truncation: Union[bool, str]
    max_length: Optional[int]
    add_special_tokens: bool
    return_attention_mask: bool
    return_token_type_ids: Optional[bool]
    return_tensors: str


class VLMDecodeKwargs(TypedDict, total=False):
    """TypedDict for VLM decode method parameters."""

    skip_special_tokens: bool
    clean_up_tokenization_spaces: bool
