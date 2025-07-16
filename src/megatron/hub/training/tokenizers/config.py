# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Literal, Optional, Union

from megatron.core.tokenizers import MegatronTokenizerBase


@dataclass
class TokenizerConfig:
    """Configuration settings for the tokenizer."""

    tokenizer_path: str = None
    """Path to tokenizer model."""

    metadata: Optional[Union[str | dict]] = None
    """Tokenizer metadata."""

    multimodal_tokenizer: Optional[bool] = False
    """Whether to use multimodal tokenizer."""

    additional_args: Optional[dict] = None
    """Tokenizer additional arguments."""

    # Multimodal tokenizer arguments
    tokenizer_prompt_format: Optional[str] = None
    special_tokens: Optional[list[str]] = None
    image_tag_type: Optional[str] = None

    # Metadata arguments
    write_metadata: Optional[bool] = True
    """Creates tokenizer metadata file."""

    tokenizer_library: Optional[
        Literal[
            "huggingface",
            "sentencepiece",
            "tiktoken",
            "megatron",
            "null",
            "byte-level",
        ]
    ] = None

    model_type: Optional[str] = None
    """Type of LLM model to be used with tokenizer."""

    tokenizer_class: Optional[MegatronTokenizerBase] = None
    """Pre-defined tokenizer class."""

    chat_template: Optional[str] = None
    """Tokenizer chat template."""

    overwrite_metadata: Optional[bool] = False
    """If overwrite metadata file."""

    metadata_path: Optional[str] = None
    """Path to save metadata file."""
