#!/usr/bin/env python3
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

"""
Qwen3 14B Continued Pretraining Script with YAML and CLI Configuration Overrides.

This script provides a flexible way to continue pretraining Qwen3 14B models using Megatron-Bridge with support for
both YAML configuration files and command-line overrides using Hydra-style syntax.

Key differences from full pretraining:
- Loads existing checkpoint by default
- Uses lower learning rate to prevent catastrophic forgetting
- Supports loading optimizer and scheduler states
- Uses 8192 sequence length for better context handling
- More frequent evaluation for monitoring continued training progress

Examples:
    Basic continued pretraining with checkpoint loading:
        $ torchrun --nproc_per_node=8 continued_pretrain_qwen3_14b.py \
        checkpoint.load=/path/to/existing/checkpoint

    Using custom data and checkpoint paths:
        $ torchrun --nproc_per_node=8 continued_pretrain_qwen3_14b.py \
        --config-file conf/qwen3_14b_continued_pretrain.yaml \
        checkpoint.load=/path/to/existing/checkpoint \
        dataset.blend=[/path/to/new/dataset] \
        train.train_iters=10000

    Adjusting learning rate for continued training:
        $ torchrun --nproc_per_node=8 continued_pretrain_qwen3_14b.py \
        checkpoint.load=/path/to/existing/checkpoint \
        optimizer.lr=5e-5 \
        optimizer.min_lr=5e-6

Configuration Precedence:
    1. Base configuration from pretrain_config() recipe
    2. YAML overrides from --config-file (defaults to conf/qwen3_14b_continued_pretrain.yaml)
    3. CLI overrides (highest precedence)

Required Parameters:
    - checkpoint.load: Path to existing checkpoint directory (required for continued pretraining)
    - dataset.blend or data_paths: Training data paths

Supported Override Syntax:
    - Standard assignment: key=value
    - Nested assignment: section.subsection.key=value
    - Addition: +new_key=value
    - Deletion: ~key_to_remove
    - Type conversion: Automatic for basic types (int, float, bool, str)
    - Complex types: torch.dtype, enums, etc. are supported
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen.qwen3_14b import pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


# Define paths relative to this script's location
SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "qwen3_14b_continued_pretrain.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Continue pretraining Qwen3 14B model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/qwen3_14b_continued_pretrain.yaml",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def validate_continued_pretraining_config(cfg: ConfigContainer) -> None:
    """
    Validate that the configuration is properly set up for continued pretraining.

    Args:
        cfg: The configuration container to validate

    Raises:
        ValueError: If required settings for continued pretraining are missing
    """
    # Check if checkpoint load path is specified
    if not cfg.checkpoint.load:
        raise ValueError(
            "checkpoint.load must be specified for continued pretraining. "
            "Please provide the path to an existing checkpoint directory."
        )

    # Check if checkpoint directory exists
    if not os.path.exists(cfg.checkpoint.load):
        raise ValueError(
            f"Checkpoint directory does not exist: {cfg.checkpoint.load}"
        )

    # Warn if learning rate seems too high for continued pretraining
    if hasattr(cfg.optimizer, 'lr') and cfg.optimizer.lr > 2e-4:
        logger.warning(
            f"Learning rate {cfg.optimizer.lr} may be too high for continued pretraining. "
            f"Consider using a lower value (e.g., 1e-4 or lower) to prevent catastrophic forgetting."
        )

    # Check data configuration
    if not (cfg.dataset.blend or hasattr(cfg.dataset, 'data_paths')):
        raise ValueError(
            "No training data specified. Please set dataset.blend or provide data_paths."
        )


def main() -> None:
    """
    Entry point for the Qwen3 14B continued pretraining script.

    This function orchestrates the complete configuration workflow for continued pretraining:
    1. Loads the base configuration from pretrain_config() recipe
    2. Applies YAML overrides from --config-file (defaults to continued pretraining config)
    3. Applies CLI overrides using Hydra-style syntax
    4. Validates the configuration for continued pretraining requirements
    5. Starts Megatron pretraining with the final merged configuration

    The script ensures that:
    - Existing checkpoints are properly loaded
    - Learning rates are appropriate for continued training
    - Sequence length is set to 8192 for better context handling
    - Optimizer and scheduler states are loaded when available

    Examples of CLI usage:
        # Basic continued pretraining
        torchrun --nproc_per_node=8 continued_pretrain_qwen3_14b.py \
            checkpoint.load=/path/to/checkpoint

        # Custom learning rate and data
        torchrun --nproc_per_node=8 continued_pretrain_qwen3_14b.py \
            checkpoint.load=/path/to/checkpoint \
            optimizer.lr=5e-5 \
            dataset.blend=[/path/to/data]

        # Extended sequence length training
        torchrun --nproc_per_node=8 continued_pretrain_qwen3_14b.py \
            checkpoint.load=/path/to/checkpoint \
            dataset.sequence_length=16384 \
            model.seq_length=16384
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Qwen3 14B Continued Pretraining Script with YAML & CLI Overrides")
    logger.info("================================================================================")

    # Load base configuration from the recipe as a Python dataclass
    # Use continued pretraining friendly defaults
    cfg: ConfigContainer = pretrain_config(
        # Default to 8192 sequence length for better context
        seq_length=8192,
        # Lower learning rate for continued training
        lr=1e-4,
        min_lr=1e-5,
        # Shorter warmup for continued training
        lr_warmup_iters=100,
        # Fewer iterations as this is continuation
        train_iters=50000,
    )
    logger.info("Loaded base configuration for continued pretraining")

    # Print configuration on rank 0
    if get_rank_safe() == 0:
        cfg.print_yaml()

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides if a config file is provided
    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.warning(f"Override YAML file not found: {args.config_file}")
            logger.info("Proceeding with base configuration and CLI overrides only.")
        else:
            yaml_overrides_omega = OmegaConf.load(args.config_file)
            merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
            logger.debug("YAML overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Validate configuration for continued pretraining
    try:
        validate_continued_pretraining_config(cfg)
        logger.info("Configuration validation passed for continued pretraining.")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Display final configuration
    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration for Continued Pretraining ---")
        cfg.print_yaml()
        logger.info("------------------------------------------------------------")

        # Additional info for continued pretraining
        logger.info(f"Loading checkpoint from: {cfg.checkpoint.load}")
        logger.info(f"Saving new checkpoints to: {cfg.checkpoint.save}")
        logger.info(f"Sequence length: {cfg.dataset.sequence_length}")
        logger.info(f"Learning rate: {getattr(cfg.optimizer, 'lr', 'N/A')}")
        logger.info(f"Training iterations: {cfg.train.train_iters}")

    # Start continued pretraining
    logger.info("Starting continued pretraining...")
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()