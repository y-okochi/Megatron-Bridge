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

import unittest.mock as mock

import pytest

from megatron.bridge.peft.lora_merge import (
    _load_full_config_container_from_checkpoint,
    _modify_checkpoint_config_for_merge,
    _prepare_merged_config,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer


class TestLoRAMerge:
    """Unit tests for LoRA merge functionality."""

    def test_modify_checkpoint_config_for_merge(self):
        """Test checkpoint config modification for merge operation."""
        original_config = CheckpointConfig(
            save="/original/save/path",
            load="/original/load/path",
            pretrained_checkpoint="/pretrained/path",
            finetune=True,
            load_optim=True,
            load_rng=True,
            ckpt_format="torch_dist",
            async_save=True,
        )

        lora_checkpoint_path = "/path/to/lora/checkpoint"

        modified_config = _modify_checkpoint_config_for_merge(original_config, lora_checkpoint_path)

        # Check that merge-specific settings are applied
        assert modified_config.load == lora_checkpoint_path
        assert modified_config.finetune == False
        assert modified_config.load_optim == False
        assert modified_config.load_rng == False

        # Check that other settings are preserved
        assert modified_config.save == "/original/save/path"
        assert modified_config.pretrained_checkpoint == "/pretrained/path"
        assert modified_config.ckpt_format == "torch_dist"
        assert modified_config.async_save == True

    def test_prepare_merged_config(self):
        """Test preparation of config for merged checkpoint."""
        # Create a mock original config
        original_checkpoint_config = CheckpointConfig(
            save="/original/save",
            load="/lora/checkpoint",
            pretrained_checkpoint="/base/model",
            finetune=False,
            load_optim=False,
            load_rng=False,
        )

        # Create a simple mock config container
        original_config = mock.MagicMock(spec=ConfigContainer)
        original_config.checkpoint = original_checkpoint_config

        output_path = "/merged/output"

        with mock.patch("megatron.bridge.peft.lora_merge.replace") as mock_replace:
            # Mock the replace calls
            mock_replace.side_effect = lambda obj, **kwargs: mock.MagicMock(
                **{**{attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith("_")}, **kwargs}
            )

            _ = _prepare_merged_config(original_config, output_path)

            # Verify replace was called correctly for checkpoint config
            checkpoint_calls = [
                call
                for call in mock_replace.call_args_list
                if "save" in call.kwargs or "pretrained_checkpoint" in call.kwargs
            ]
            assert len(checkpoint_calls) >= 1

            # Check that expected modifications were requested
            found_checkpoint_update = False
            for call in checkpoint_calls:
                if call.kwargs.get("save") == output_path:
                    found_checkpoint_update = True
                    assert call.kwargs.get("pretrained_checkpoint") is None
                    assert call.kwargs.get("ckpt_format") == "torch_dist"

            assert found_checkpoint_update, "Checkpoint config not updated correctly"

    @mock.patch("megatron.bridge.training.checkpointing.get_checkpoint_run_config_filename")
    @mock.patch("megatron.bridge.training.utils.checkpoint_utils.file_exists")
    @mock.patch("megatron.bridge.training.checkpointing.read_run_config")
    @mock.patch("megatron.bridge.utils.instantiate_utils.instantiate")
    def test_load_full_config_container_from_checkpoint(
        self, mock_instantiate, mock_read_config, mock_file_exists, mock_get_filename
    ):
        """Test loading full config container from checkpoint."""
        # Setup mocks
        lora_checkpoint_path = "/path/to/lora/checkpoint"
        config_filename = "/path/to/lora/checkpoint/run_config.yaml"

        mock_get_filename.return_value = config_filename
        mock_file_exists.return_value = True

        # Mock run config data
        mock_run_config = {
            "train": {"_target_": "TrainingConfig", "train_iters": 100},
            "model": {"_target_": "GPTModelProvider", "num_layers": 12},
            "optimizer": {"_target_": "OptimizerConfig", "lr": 1e-4},
            "scheduler": {"_target_": "SchedulerConfig", "lr_decay_style": "cosine"},
            "dataset": {"_target_": "DatasetConfig", "seq_length": 512},
            "logger": {"_target_": "LoggerConfig", "log_interval": 10},
            "tokenizer": {"_target_": "TokenizerConfig", "vocab_size": 50000},
            "ddp": {"_target_": "DDPConfig"},
            "dist": {"_target_": "DistConfig"},
            "checkpoint": {"_target_": "CheckpointConfig", "save_interval": 100},
            "peft": {"_target_": "LoRA", "dim": 16},
            "rng": {"_target_": "RNGConfig", "seed": 1234},
        }
        mock_read_config.return_value = mock_run_config

        # Mock instantiate to return mock objects
        def mock_instantiate_fn(config_dict):
            return mock.MagicMock(spec_name=config_dict.get("_target_", "MockConfig"))

        mock_instantiate.side_effect = mock_instantiate_fn

        with mock.patch("megatron.bridge.peft.lora_merge._modify_checkpoint_config_for_merge") as mock_modify:
            mock_modify.return_value = mock.MagicMock()

            # Call the function
            _ = _load_full_config_container_from_checkpoint(lora_checkpoint_path)

            # Verify file operations
            mock_get_filename.assert_called_once_with(lora_checkpoint_path)
            mock_file_exists.assert_called_once_with(config_filename)
            mock_read_config.assert_called_once_with(config_filename)

            # Verify all config sections were instantiated
            expected_calls = len(mock_run_config)  # All config sections
            assert mock_instantiate.call_count >= expected_calls - 2  # Allow for optional configs

            # Verify checkpoint config was modified
            mock_modify.assert_called_once()

    @mock.patch("megatron.bridge.training.utils.checkpoint_utils.file_exists")
    def test_load_config_missing_file(self, mock_file_exists):
        """Test error handling when run_config.yaml is missing."""
        mock_file_exists.return_value = False

        with pytest.raises(ValueError, match="Run config not found"):
            _load_full_config_container_from_checkpoint("/nonexistent/checkpoint")
