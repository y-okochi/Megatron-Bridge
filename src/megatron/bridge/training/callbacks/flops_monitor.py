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

from megatron.bridge.training.metrics.abstract_monitor import AbstractMonitor

_model_flops_map = {
    "gpt3": flops_formulas.gpt3,
    "llama2": flops_formulas.llama2,
    "llama3": flops_formulas.llama3,
    "llama4": flops_formulas.llama3,  # TODO: add llama4 flops formulas
    "nemotron3": flops_formulas.nemotron,
    "nemotron4": flops_formulas.nemotron,
    "mixtral": flops_formulas.mixtral,
    "bert": flops_formulas.bert,
    "hyena": hyena,
    "deepseekv3": flops_formulas.deepseekv3,
    "transformer": flops_formulas.transformer,
    "qwen3": flops_formulas.qwen3,
    "nemotronh": flops_formulas.nemotronh,
    "gpt_oss": flops_formulas.gpt_oss,
}


class FLOPsMonitor(AbstractMonitor):
    """
    Calculate and log FLOPs per second after every ``trainer.log_every_n_steps`` steps.

    Args:
        model_config (GPTConfig): Model parameters.
        data_config (pl.LightningDataModule): Data module being used in the experiment.
        model_name (str): Name of the model being run. The following models are supported:
            gpt3, llama2, llama3, nemotron, mixtral, bert, hyena.


    """

    higher_is_better = True

    def __init__(
        self,
        model_config: GPTConfig,
        data_config: pl.LightningDataModule,
        model_name: str,
    ):
        self.model_cfg = model_config
        self.data_cfg = data_config

        # use config params only when NOT provided explicitly
        self.model = model_name

        gbs = self.data_cfg.global_batch_size
        enc_seq_len = self.model_cfg.seq_length
        hs = self.model_cfg.hidden_size
        layers = self.model_cfg.num_layers
        ffn_hs = self.model_cfg.ffn_hidden_size
        attention_heads = self.model_cfg.num_attention_heads
        moe_router_topk = self.model_cfg.moe_router_topk
        model_pattern = getattr(self.model_cfg, "hybrid_override_pattern", None)
        vocab_size = self.data_cfg.tokenizer.vocab_size if hasattr(self.data_cfg, "tokenizer") else None

        # this handles both- 1. key is present, value is None; 2. key is absent
        query_groups = self.model_cfg.num_query_groups
        if query_groups is None:
            query_groups = attention_heads

        config_kwargs = {
            "gbs": gbs,
            "enc_seq_len": enc_seq_len,
            "hs": hs,
            "layers": layers,
            "ffn_hs": ffn_hs,
            "attention_heads": attention_heads,
            "moe_router_topk": moe_router_topk,
            "query_groups": query_groups,
            "vocab_size": vocab_size,
            "model_pattern": model_pattern,
        }

        from megatron.core.transformer.transformer_config import MLATransformerConfig

        if isinstance(self.model_cfg, MLATransformerConfig):
            config_kwargs["qk_head_dim"] = self.model_cfg.qk_head_dim
            config_kwargs["qk_pos_emb_head_dim"] = self.model_cfg.qk_pos_emb_head_dim
            config_kwargs["v_head_dim"] = self.model_cfg.v_head_dim
            config_kwargs["q_lora_rank"] = self.model_cfg.q_lora_rank
            config_kwargs["kv_lora_rank"] = self.model_cfg.kv_lora_rank
        config_kwargs["moe_layer_freq"] = self.model_cfg.moe_layer_freq
        config_kwargs["moe_shared_expert_intermediate_size"] = self.model_cfg.moe_shared_expert_intermediate_size
        config_kwargs["moe_ffn_hidden_size"] = self.model_cfg.moe_ffn_hidden_size
        config_kwargs["mtp_num_layers"] = self.model_cfg.mtp_num_layers

        if self.model_cfg.is_hybrid_model:
            config_kwargs['is_hybrid_model'] = True
            config_kwargs['hybrid_override_pattern'] = self.model_cfg.hybrid_override_pattern
            config_kwargs['mamba_state_dim'] = self.model_cfg.mamba_state_dim
            config_kwargs['mamba_head_dim'] = self.model_cfg.mamba_head_dim
            config_kwargs['mamba_num_groups'] = self.model_cfg.mamba_num_groups
            config_kwargs['mamba_num_heads'] = self.model_cfg.mamba_num_heads

        if self.model_cfg.window_size is not None:
            config_kwargs["window_size"] = self.model_cfg.window_size
        if getattr(self.model_cfg, "window_attn_skip_freq", None) is not None:
            config_kwargs["window_attn_skip_freq"] = self.model_cfg.window_attn_skip_freq
        if self.model_cfg.kv_channels is not None:
            config_kwargs["kv_channels"] = self.model_cfg.kv_channels

        self.flops_config = flops_formulas.FLOPSConfig(**config_kwargs)

        self.model = self.model.lower() if self.model is not None else self.model

        self.avg_train_step_time = 0

    def on_train_start(self, trainer, pl_module):
        """
        PyTorch Lightning callback hook. Ensures that user is not using PEFT
        as FLOPS callback does not support it.
        """
        for callback in trainer.callbacks:
            if isinstance(callback, PEFT):
                raise NotImplementedError("FLOPs measurement not supported for finetuning jobs")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        """
        PyTorch Lightning callback hook to calculate TFLOPs per sec per GPU after training
        """
        try:
            self.avg_train_step_time += trainer.progress_bar_metrics['train_step_timing in s']
        except KeyError:
            print("'train_step_timing in s' not found. Make sure to use TimingCallback with FLOPsMeasurementCallback.")
        n = trainer.strategy.current_epoch_step
        if n % trainer.log_every_n_steps == 0:
            # skip calculation if we haven't accumulated any timing data
            if self.avg_train_step_time == 0:
                return
            train_step_time = self.avg_train_step_time / trainer.log_every_n_steps
            tflops_per_gpu, flops = self.eval_tflops_per_sec_per_gpu(train_step_time)
            self.avg_train_step_time = 0
            pl_module.log(
                "TFLOPS_per_GPU",
                tflops_per_gpu,
                on_step=True,
                on_epoch=False,
                batch_size=1,
                prog_bar=True,
            )

            tflops = flops / (1e12 * train_step_time)
            pl_module.log(
                "TFLOPS",
                tflops,
            )

    def eval_tflops_per_sec_per_gpu(self, train_step_time: List | float | int) -> float:
        """
        Args:
            train_step_time (Any[List, float, int]): Train step time (in seconds).
            Step time will be less stable for initial steps (~10 steps)- less
            accurate measurement
            Use average step time over several steps for higher accuracy
        Returns:
            (float): Model TFLOPs per sec per gpu
        """
        total_flops, flops_per_gpu = self.eval_model_flops()

        if not isinstance(train_step_time, list):
            train_step_time = [train_step_time]
        # efficient mean computation if num train steps is very large
        step_time_arr = np.array(train_step_time)
        train_step_time = np.mean(step_time_arr[len(step_time_arr) // 2 :])

        flops_per_sec_per_gpu = flops_per_gpu / (1e12 * train_step_time)

        return flops_per_sec_per_gpu, total_flops

    def eval_model_flops(self) -> Tuple[float, float]:
        """
        Calculate model FLOPs for a given model
        """
        if self.model is not None:
            model_matches = [model for model in _model_flops_map if model in self.model]
            self.model = model_matches[0] if len(model_matches) > 0 else self.model
        if self.model not in _model_flops_map:
            logging.info(f"FLOPs measurement supported for {list(_model_flops_map.keys())}")
            raise KeyError(f"Failed to extract valid model name from or missing FLOPs calculations for {self.model}")

        total_flops = _model_flops_map[self.model](self.flops_config)
        num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        flops_per_gpu = total_flops / num_devices

        return total_flops, flops_per_gpu