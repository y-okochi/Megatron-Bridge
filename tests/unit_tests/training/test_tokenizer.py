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

from megatron.hub.training.tokenizers import build_tokenizer
from megatron.hub.training.tokenizers.config import TokenizerConfig


class TestTokenizerConfig:
    def test_tokenizer_config(self):
        additional_args = {
            "num_special_tokens": 100,
            "pattern": "v1",
        }
        config = TokenizerConfig(
            tokenizer_path="/path/to/model",
            metadata_path=dict(library="tiktoken"),
            additional_args=additional_args,
        )

        assert config.metadata_path == {"library": "tiktoken"}
        assert config.additional_args["pattern"] == "v1"

    def test_build_tokenizer_multimodal(self):
        special_tokens = ["<image_start>", "<image_end>"]
        config = TokenizerConfig(
            tokenizer_path="llava-hf/llava-1.5-7b-hf",
            multimodal_tokenizer=True,
            tokenizer_prompt_format="nvlm-yi-34b",
            special_tokens=special_tokens,
            image_tag_type="nvlm",
        )

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

    def test_build_tokenzier_megatron(self):
        additional_args = {}
        additional_args["additional_special_tokens"] = [f"<extra_id_{i}>" for i in range(100)]
        config = TokenizerConfig(
            tokenizer_path="BertWordPieceCase",
            metadata_path=dict(library="megatron"),
            additional_args=additional_args,
        )

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

    def test_build_tokenizer_huggingface(self):
        config = TokenizerConfig(
            tokenizer_path="nvidia/Minitron-4B-Base",
            metadata_path=dict(library="huggingface"),
        )

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )

    def test_build_tokenizer_null(self):
        additional_args = dict(vocab_size=131072)
        config = TokenizerConfig(
            metadata_path=dict(library="null"),
            additional_args=additional_args,
        )

        tokenizer = build_tokenizer(
            tokenizer_config=config,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )
