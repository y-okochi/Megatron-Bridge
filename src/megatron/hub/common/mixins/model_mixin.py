import abc
from typing import Generic, TypeVar
import os


import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.hub.core.models.model_provider import get_model
from megatron.hub.common.mixins.config_mixin import ConfigMixin

ModelT = TypeVar("ModelT", bound=MegatronModule)


class ModelProviderMixin(abc.ABC, Generic[ModelT], ConfigMixin):
    CONFIG_NAME = "mhub_model.json"

    @abc.abstractmethod
    def provide(self, pre_process=None, post_process=None) -> ModelT:
        pass

    def get_model(
        self,
        ddp_config: DistributedDataParallelConfig | None = None,
        model_type=ModelType.encoder_or_decoder,
        overlap_param_gather_with_optimizer_step: bool = False,
        fp16: bool | None = None,
        bf16: bool | None = None,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        use_cpu_initialization: None | bool = False,
        init_model_with_meta_device: bool | None = None,
    ) -> list[ModelT]:
        """Get a wrapped and configured GPT model ready for training.

        Args:
            ddp_config: Configuration for distributed data parallel
            model_type: Type of model (encoder, decoder, or both)
            overlap_param_gather_with_optimizer_step: Whether to overlap param gathering
            fp16: Override FP16 setting
            bf16: Override BF16 setting
            use_torch_fsdp2: Use PyTorch FSDP2 instead of custom DDP
            wrap_with_ddp: Whether to wrap model with DDP
            data_parallel_random_init: Initialize parameters randomly across data parallel ranks
            use_cpu_initialization: Initialize model on CPU

        Returns:
            List of wrapped model modules
        """
        if wrap_with_ddp and not ddp_config:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        if not torch.distributed.is_initialized():
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            torch.distributed.init_process_group("nccl")
        
        if not parallel_state.is_initialized():
            print("Model parallel not initialized, initializing...")
            self.initialize_model_parallel(seed=0)

        return get_model(
            self.provide,
            ddp_config=ddp_config,
            model_type=model_type,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            fp16=fp16,
            bf16=bf16,
            use_torch_fsdp2=use_torch_fsdp2,   
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            use_cpu_initialization=use_cpu_initialization,
            init_model_with_meta_device=init_model_with_meta_device,
        )
    
    def initialize_model_parallel(self, seed: int | None = None, **kwargs) -> None:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
            torch.cuda.set_device(torch.distributed.get_rank())
        
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=getattr(self, "tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=getattr(self, "pipeline_model_parallel_size", 1),
            virtual_pipeline_model_parallel_size=getattr(self, "virtual_pipeline_model_parallel_size", None),
            context_parallel_size=getattr(self, "context_parallel_size", 1) or 1,
            expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1) or 1,
            **kwargs,
        )
        if seed is not None:
            model_parallel_cuda_manual_seed(seed)

    def __call__(
        self,
        ddp_config: DistributedDataParallelConfig | None = None,
        model_type=ModelType.encoder_or_decoder,
        overlap_param_gather_with_optimizer_step: bool = False,
        fp16: bool | None = None,
        bf16: bool | None = None,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        use_cpu_initialization: None | bool = False,
        init_model_with_meta_device: bool | None = None,
    ) -> list[ModelT]:
        return self.get_model(
            ddp_config=ddp_config,
            model_type=model_type,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            fp16=fp16,
            bf16=bf16,
            use_torch_fsdp2=use_torch_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            use_cpu_initialization=use_cpu_initialization,
            init_model_with_meta_device=init_model_with_meta_device,
        )

    @property
    def meta_model(self) -> list[ModelT]:
        return self(wrap_with_ddp=False, init_model_with_meta_device=True)
