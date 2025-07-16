# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from megatron.core.tokenizer import MegatronTokenizer

from megatron.hub.core.utils.common_utils import get_rank_safe
from megatron.hub.training.tokenizers.config import TokenizerConfig
from megatron.hub.training.tokenizers.multimodal_tokenizer import MultimodalTokenizer


def build_tokenizer(
    tokenizer_config: TokenizerConfig, make_vocab_size_divisible_by: int, tensor_model_parallel_size: int, **kwargs
):
    """Initialize tokenizer based on the provided configuration.

    This function serves as a factory to instantiate various tokenizer types
    supported by NeMo, such as BERT, GPT2, SentencePiece, HuggingFace, etc.
    It also handles padding the vocabulary size to be GPU-friendly.

    Args:
        tokenizer_config (TokenizerConfig): Configuration object specifying the tokenizer
                                            type, paths to vocab/model files, and other
                                            tokenizer-specific settings.
        make_vocab_size_divisible_by (int): Ensures the vocabulary size is a multiple of this value.
        tensor_model_parallel_size (int): The tensor model parallel size, used for further
                                          adjusting vocabulary size for distributed training.
        **kwargs: Additional keyword arguments that might be specific to certain tokenizers
                  (e.g., passed to HuggingFace AutoTokenizer).

    Returns:
        MegatronTokenizer: An instance of the initialized tokenizer.

    Raises:
        NotImplementedError: If the specified tokenizer_type in tokenizer_config is not supported.
        ImportError: If a required library (e.g., transformers for MultimodalTokenizer) is not installed.
    """
    if get_rank_safe() == 0:
        print("> building {} tokenizer ...".format(tokenizer_config.tokenizer_type), flush=True)

    if tokenizer_config.multimodal_tokenizer:
        try:
            import transformers as _transformers
        except ImportError as exc:
            raise ImportError("MultimodalTokenizer currently requires transformers library to be installed") from exc
        kwargs = {}
        if tokenizer_config.tokenizer_prompt_format == "nvlm-yi-34b":
            kwargs = {"from_slow": True, "legacy": False, "add_bos_token": True}
        underlying_tokenizer = _transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_config.tokenizer_model, **kwargs
        )
        tokenizer = MultimodalTokenizer(
            underlying_tokenizer,
            tokenizer_config.tokenizer_prompt_format,
            tokenizer_config.special_tokens,
            tokenizer_config.image_tag_type,
        )
    else:
        if tokenizer_config.write_metadata:
            MegatronTokenizer.write_metadata(
                tokenizer_path=tokenizer_config.tokenizer_path,
                tokenizer_library=tokenizer_config.tokenizer_library,
                model_type=tokenizer_config.model_type,
                tokenizer_class=tokenizer_config.tokenizer_class,
                overwrite=tokenizer_config.overwrite_metadata,
                metadata_path=tokenizer_config.metadata_path,
            )
        
        tokenizer = MegatronTokenizer.from_pretrained(
            tokenizer_path=tokenizer_config.tokenizer_path,
            metadata_path=tokenizer_config.metadata,
            **tokenizer_config.additional_args,
        )

    return tokenizer
