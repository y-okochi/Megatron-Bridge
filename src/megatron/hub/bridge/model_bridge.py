"""
This module provides a framework for seamlessly converting models between
Hugging Face (`transformers`) and Megatron-Core formats.

It is designed to be extensible, allowing developers to easily add support for new
model architectures by creating a custom "bridge".

For a detailed explanation of the design and a guide on how to add new model
bridges, please see `docs/bridge/model_bridge.md`.
"""
import abc
from typing import Generic, TypeVar, Type, Callable, List, Tuple, Iterable, Optional, NamedTuple, Mapping, Literal, Dict, Union
from functools import partial
from dataclasses import dataclass
from collections import defaultdict


import torch
from transformers.modeling_utils import PreTrainedModel
from megatron.core.transformer.module import MegatronModule
from megatron.core import parallel_state as mpu
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


from megatron.hub.core.models.model_provider import ModelProviderProtocol
from megatron.hub.bridge.state_bridge import MegatronStateBridge
from megatron.hub.bridge.weight_bridge import MegatronWeightBridge
from megatron.hub.common.decorators import dispatch


WeightBridgeT = TypeVar("WeightBridgeT", bound=MegatronWeightBridge)


class MegatronWeightTuple(NamedTuple):
    model_idx: int
    param_name: str
    weight: torch.Tensor


class HFWeightTuple(NamedTuple):
    param_name: str
    weight: torch.Tensor


class HFSaveTask(NamedTuple):
    pp_rank: int
    vpp_rank: Optional[int]
    megatron_name: str
    weight_bridge: MegatronWeightBridge


# ---------------------------------------------------------------------------
# Lightweight immutable task records used by the new orchestration logic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HFLoadTask(Generic[WeightBridgeT]):
    """A single step in the *HF ➜ Megatron* loading schedule.

    Attributes
    ----------
    model_idx
        Index into the `dst` list passed to :py:meth:`MegatronModelBridge.load_state_from_hf`.
    param_name
        Fully-qualified, *unwrapped* Megatron parameter name (no ``module.`` prefixes).
    megatron_module
        Reference to the Megatron model (or sub-module) that **owns** the parameter.  
        Needed by the :pyclass:`~mhub.hub.bridge.weight_bridge.MegatronWeightBridge` for
        configuration information (e.g. hidden size, number of heads).
    megatron_param
        The actual :pyclass:`torch.nn.Parameter` object which will receive the shard on
        *this* process after the bridge finishes any TP/PP communication.
    bridge
        Concrete :pyclass:`MegatronWeightBridge` instance responsible for all heavy-lifting
        (format conversion, TP scatter, PP broadcast).
    """
    model_idx: int
    param_name: str
    megatron_module: torch.nn.Module
    megatron_param: torch.Tensor
    bridge: WeightBridgeT

    def to_megatron(
        self,
        weights: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward to the underlying bridge's `to_megatron`."""
        return self.bridge.to_megatron(weights, megatron_module)


@dataclass(frozen=True)
class _HFSaveTask(Generic[WeightBridgeT]):
    """A single step in the *Megatron ➜ HF* export schedule.

    Attributes
    ----------
    pp_rank
        Pipeline-parallel rank that **owns** the parameter.
    vpp_rank
        Virtual-pipeline (intra-layer) rank if model is built with VPP, otherwise ``None``.
    param_name
        Fully-qualified, *unwrapped* Megatron parameter name to export.
    bridge
        :pyclass:`MegatronWeightBridge` instance which will gather TP shards,
        broadcast from the owning PP rank, perform any reshaping, and finally
        return a ``dict[str, Tensor]`` mapping HF keys → tensors.
    """

    pp_rank: int
    vpp_rank: Optional[int]
    param_name: str
    bridge: WeightBridgeT

    def from_megatron(
        self,
        megatron_weight: Optional[torch.Tensor],
        megatron_module: Optional[torch.nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Forward to the underlying bridge's `from_megatron`."""
        return self.bridge.from_megatron(megatron_weight, megatron_module)


HFPreTrained = TypeVar("HFPreTrained")
ModelProviderTarget = TypeVar("ModelProviderTarget", bound=ModelProviderProtocol)
MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
_BridgeImplClass = TypeVar("_BridgeImplClass", bound="MegatronModelBridge")


@dispatch
def to_megatron(hf_architecture, hf_pretrained, load_weights: bool = True) -> ModelProviderProtocol:
    ...


@dispatch
def bridge_state_to_megatron(hf_architecture, hf_pretrained) -> Iterable[MegatronWeightTuple]:
    ...


@dispatch
def bridge_state_to_hf(
    first_model: MegatronModel,
    model: list[MegatronModel], 
    hf_pretrained: HFPreTrained, 
    cpu: bool = True, 
    order: Literal["megatron", "hf", "safetensors"] = "safetensors",
    show_progress: bool = True
) -> Iterable[HFWeightTuple]:
    ...



class MegatronModelBridge(Generic[HFPreTrained, ModelProviderTarget, MegatronModel]):
    """High-level *orchestrator* that wires HuggingFace and Megatron together.

    The class itself **does not** understand any tensor-parallel or pipeline-parallel
    details – all such knowledge lives in concrete
    :pyclass:`~mhub.hub.bridge.weight_bridge.MegatronWeightBridge` subclasses.

    What this class *does* is:

    1. Build deterministic *plans* (lists of :class:`_HFLoadTask` / :class:`_HFSaveTask`)
       that answer *"which parameter is handled by which bridge?"*  Nothing more.
    2. Walk those plans, calling :py:meth:`MegatronWeightBridge.to_megatron` or
       :py:meth:`MegatronWeightBridge.from_megatron` as appropriate.
    3. Provide progress bars and a small set of utility helpers for subclasses.

    To add a new model architecture you typically:

    • Subclass :class:`MegatronModelBridge`.
    • Implement :py:meth:`provider_bridge` – returns a Megatron *model provider*.
    • Implement :py:meth:`state_bridge`   – returns a :class:`MegatronStateBridge`
      that contains all weight mappings for the model.
    """
    @classmethod
    def impl(
        cls,
        *,
        source: Type[PreTrainedModel],
        target: Type[MegatronModel]
    ) -> Callable[[_BridgeImplClass], _BridgeImplClass]:
        """
        A class decorator factory for registering bridge implementations with
        to_megatron and from_megatron dispatchers.
        """
        def decorator(decorated_class: _BridgeImplClass) -> _BridgeImplClass:
            decorated_class_name = decorated_class.__name__

            @to_megatron.impl(source)
            def _to_megatron_registered_impl(_, hf_pretrained: HFPreTrained, load_weights: bool = True) -> ModelProviderTarget:
                bridge = decorated_class()
                provider = bridge.provider_bridge(hf_pretrained)

                if load_weights:
                    provider.model_transform = partial(bridge.load_state_from_hf, hf_pretrained)
                
                return provider
            

            @bridge_state_to_megatron.impl(source)
            def _bridge_weights_impl(_, hf_pretrained: HFPreTrained) -> Iterable[MegatronWeightTuple]:
                bridge = decorated_class()
                return bridge.bridge_state_from_hf(hf_pretrained)
            

            @bridge_state_to_hf.impl((source, target))
            def _from_megatron_registered_impl(
                _,
                megatron_models: List[MegatronModel], 
                hf_pretrained: HFPreTrained,
                cpu: bool = True,
                order: Literal["megatron", "hf", "safetensors"] = "safetensors",
                show_progress: bool = True
            ) -> Iterable[HFWeightTuple]:
                bridge = decorated_class()
                return bridge.bridge_state_to_hf(megatron_models, hf_pretrained, cpu=cpu, order=order, show_progress=show_progress)
            
            
            _to_megatron_registered_impl.__name__ = f"_to_megatron_with_{decorated_class_name}"
            _bridge_weights_impl.__name__ = f"_bridge_weights_with_{decorated_class_name}"
            _from_megatron_registered_impl.__name__ = f"_from_megatron_with_{decorated_class_name}"

            return decorated_class
        return decorator
    
    @abc.abstractmethod
    def provider_bridge(self, hf_pretrained: HFPreTrained) -> ModelProviderTarget:
        raise NotImplementedError("Subclass must implement bridge method")

    @abc.abstractmethod
    def state_bridge(self) -> MegatronStateBridge:
        raise NotImplementedError("Subclass must implement state_bridge method")
    
    def load_state_from_hf(self, src: HFPreTrained, dst: list[MegatronModel]) -> list[MegatronModel]:
        """Top-level orchestrator for HF ➜ Megatron conversion.

        Every heavy-lifting step (format reshaping, TP/PP communication, etc.)
        is completely delegated to the MegatronWeightBridge selected for each
        parameter.  This method merely builds the plan and executes it with a
        progress bar on rank 0.
        """

        load_plan = list(self._build_plan_from_hf(src, dst))

        state_accessor: Mapping[str, torch.Tensor] = (
            src.state if hasattr(src, "state") else {}
        )

        is_main_rank = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        bridge_name = self.__class__.__name__

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[bridge]}"),
            disable=not is_main_rank,
        ) as progress:
            task_id = progress.add_task(
                f"Loading from {src.model_name_or_path}", total=len(load_plan), bridge=bridge_name
            )

            for task in load_plan:
                # 1) Fetch source tensor(s) from HF state dict
                if isinstance(task.bridge.to, str):
                    megatron_weights = state_accessor[task.bridge.to]
                else:
                    megatron_weights = {k: state_accessor[v] for k, v in task.bridge.to.items()}

                # 2) Delegate conversion & distribution to the bridge
                weight_local = task.to_megatron(megatron_weights, task.megatron_module)

                # 3) Copy into Megatron param if this rank received a shard
                if weight_local is not None:
                    # Check shape compatibility before copying
                    if weight_local.shape != task.megatron_param.shape:
                        raise ValueError(
                            f"Shape mismatch for {task.bridge.megatron}:\n"
                            f"  Expected shape: {task.megatron_param.shape}\n"
                            f"  Got shape: {weight_local.shape}\n"
                            f"  Bridge type: {type(task.bridge).__name__}\n"
                            f"  HF mapping: {task.bridge.to}"
                        )
                    task.megatron_param.data.copy_(weight_local)

                progress.update(task_id, advance=1)

        return dst
    
    def bridge_state_from_hf(self, src: HFPreTrained, dst: List[MegatronModel]) -> Iterable[MegatronWeightTuple]:
        """Generator variant of `load_state_from_hf` – yields transformed shards."""

        for task in self._build_plan_from_hf(src, dst):
            accessor: Mapping[str, torch.Tensor] = src.state
            if isinstance(task.bridge.to, str):
                src_weights = accessor[task.bridge.to]
            else:
                src_weights = {k: accessor[v] for k, v in task.bridge.to.items()}

            shard = task.to_megatron(src_weights, task.megatron_module)
            if shard is not None:
                yield MegatronWeightTuple(task.model_idx, task.param_name, shard)
    
    def bridge_state_to_hf(
        self,
        src: List[MegatronModel],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        order: Literal["megatron", "hf", "safetensors"] = "safetensors",
        show_progress: bool = True,
    ) -> Iterable[HFWeightTuple]:
        """Export Megatron weights ➜ HF format.  Bridges handle everything."""

        save_plan = list(self._build_plan_to_hf(src, hf_pretrained, order))

        is_main_rank = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        bridge_name = self.__class__.__name__

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[bridge]}"),
            disable=not (is_main_rank and show_progress),
        ) as progress:
            task_id = progress.add_task("Converting to HuggingFace", total=len(save_plan), bridge=bridge_name)

            for task in save_plan:
                # Owns param? fetch weight & module; otherwise None (bridge will broadcast)
                weight = None
                module = None
                if task.pp_rank == mpu.get_pipeline_model_parallel_rank():
                    module, weight = self._get_param_and_module_from_vpp(src, task.vpp_rank, task.param_name)

                kv_pairs = task.from_megatron(weight, module)

                for name, tensor in kv_pairs.items():
                    yield HFWeightTuple(name, tensor.cpu() if cpu else tensor)

                progress.update(task_id, advance=1)

    def dtype_from_hf(self, config):
        """
        Extracts torch dtype from a HF config
        """
        assert hasattr(config, 'torch_dtype'), "Expected config to have attr `torch_dtype`"
        torch_dtype = config.torch_dtype
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        elif isinstance(torch_dtype, str):
            return self.dtype_from_str(torch_dtype)
        else:
            raise ValueError("torch_dtype is not of type str/torch.dtype")
        
    def dtype_from_str(self, dtype):
        """
        Convert a str precision to equivalent torch dtype.
        """
        assert isinstance(dtype, str)
        if dtype in ["float16", "fp16", "16", "16-mixed"]:
            return torch.float16
        elif dtype in ["bfloat16", "bf16-mixed"]:
            return torch.bfloat16
        else:
            return torch.float32
        
    def make_vocab_size_divisible_by(self, vocab_size):
        base = 128
        while vocab_size % base != 0:
            base //= 2
        return base
    
    def _get_provider_from_model(self, model: MegatronModule) -> ModelProviderTarget:
        """Extract provider/config from model."""
        model = self._unwrap_model(model)
        return model.config
    
    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap model from DDP or other wrappers."""
        while hasattr(model, 'module'):
            model = model.module
        return model
    
    def _unwrap_name(self, name: str) -> str:
        """Unwrap name from DDP or other wrappers."""
        while name.startswith("module."):
            name = name[len("module."):]
        return name
    
    def _load_and_distribute_hf_weight(
        self,
        param: torch.Tensor,
        hf_state: Mapping[str, torch.Tensor],
        weight_bridge: MegatronWeightBridge,
        provider: ModelProviderTarget,
    ) -> Optional[torch.Tensor]:
        """
        Loads a weight from HuggingFace state on TP rank 0 and distributes it.
        """
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_world_size = mpu.get_tensor_model_parallel_world_size()

        # Load on rank 0
        if tp_rank == 0:
            weight = self._load_from_hf(hf_state, weight_bridge, provider)
            if weight is not None:
                weight = weight.to(torch.cuda.current_device())
        else:
            weight = None

        if tp_world_size == 1:
            if weight is None:
                return None
            return weight

        tp_group = mpu.get_tensor_model_parallel_group()
        tp_src_rank = mpu.get_tensor_model_parallel_src_rank()

        # On rank 0, prepare a list of tensors to scatter.
        # For other ranks, this will be None.
        if tp_rank == 0:
            is_tp = is_tensor_parallel(param)

            # If it's a tensor-parallel parameter with a specific split strategy, create shards.
            if is_tp and weight_bridge.tp_split_strategy:
                shards = weight_bridge.tp_split_strategy(provider, weight)
            else:
                # Otherwise, this is a replicated parameter (either non-TP, or TP without a split strategy).
                shards = [weight] * tp_world_size
        else:
            shards = None

        # Scatter the prepared tensors. This single operation handles all cases.
        output_weight = torch.empty_like(param, device=torch.cuda.current_device())
        torch.distributed.scatter(output_weight, shards, src=tp_src_rank, group=tp_group)
        return output_weight

    def _load_from_hf(self, hf_state: Mapping[str, torch.Tensor], weight_bridge: MegatronWeightBridge, provider: ModelProviderTarget) -> Optional[torch.Tensor]:
        """Load weight from HF using mapping."""
        if isinstance(weight_bridge.dst, str):
            # Simple mapping
            if weight_bridge.dst not in hf_state:
                return None
            weight = hf_state[weight_bridge.dst]
            # Apply to_target transformation if defined
            if weight_bridge.to_target:
                weight = weight_bridge.to_target(provider, weight)
            return weight
        else:
            # Complex mapping with transformation
            weights = {}
            for key, hf_name in weight_bridge.dst.items():
                if hf_name not in hf_state:
                    return None
                weights[key] = hf_state[hf_name]
            
            # Apply transformation
            if weight_bridge.to_target:
                return weight_bridge.to_target(provider, **weights)
            return None
    
    def _get_layer_offset(self, model: MegatronModule, vpp_idx: int) -> int:
        """Get layer offset for VPP models."""
        # TODO: Implement VPP layer offset calculation
        return 0
    
    def _adjust_name_for_vpp(self, name: str, layer_offset: int) -> str:
        """Adjust parameter name for VPP global view."""
        # TODO: Implement VPP name adjustment
        return name
    
    def _collect_all_params(self, models: List[MegatronModule]) -> List[Tuple[int, Optional[int], str]]:
        """Collect all parameter names across PP/VPP ranks."""
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        
        # Collect local names
        local_names = []
        for vpp_rank, model in enumerate(models):
            for name, _ in model.named_parameters():
                local_names.append((pp_rank, vpp_rank if len(models) > 1 else None, name))
        
        # All-gather across PP ranks
        all_names = [None] * mpu.get_pipeline_model_parallel_world_size()
        torch.distributed.all_gather_object(
            all_names, local_names, 
            group=mpu.get_pipeline_model_parallel_group()
        )
        
        # Flatten
        return sum(all_names, [])
    
    def _get_param_and_module_from_vpp(self, models: List[MegatronModule], vpp_rank: Optional[int], param_name: str) -> Optional[Tuple[torch.nn.Module, torch.Tensor]]:
        """
        Get parameter from specific VPP rank, ensuring that parameter
        attributes are preserved.
        """
        if vpp_rank is None:
            model = models[0]
        else:
            if vpp_rank >= len(models):
                return None
            model = models[vpp_rank]

        # getattr traversal ensures that we get the Parameter object, not just the
        # tensor data. This preserves attributes like `tensor_model_parallel`.
        try:
            param = self._unwrap_model(model)
            module = param
            splitted_name = param_name.split(".")
            for i, part in enumerate(splitted_name):
                param = getattr(param, part)
                if i < len(splitted_name) - 1:
                    module = getattr(module, part)
            return module, param
        except AttributeError:
            # Some params might not be found, e.g. _extra_state.
            return None
    
    def _adjust_layer_number_for_global(self, model: MegatronModule, param_name: str) -> str:
        """Adjust layer number from local to global numbering."""
        # TODO: Implement global layer number adjustment for VPP
        return param_name

    def _build_plan_from_hf(
        self, hf_src: HFPreTrained, megatron_models: list[MegatronModel]
    ) -> Iterable[_HFLoadTask]:
        """Construct the *HF ➜ Megatron* load plan.

        The algorithm walks over every parameter of every destination model,
        asks the :class:`MegatronStateBridge` whether it has a mapping for that
        parameter, and – if the corresponding HF weights actually exist – yields
        an :class:`_HFLoadTask` describing exactly how that parameter will be
        populated.
        """

        state_bridge = self.state_bridge()
        state_accessor = hf_src.state if hasattr(hf_src, "state") else {}

        for model_idx, model in enumerate(megatron_models):
            layer_offset = self._get_layer_offset(model, model_idx)
            for name, param in model.named_parameters():
                if "_extra_state" in name:
                    continue

                global_name = self._adjust_name_for_vpp(name, layer_offset)
                unwrapped = self._unwrap_name(global_name)
                bridge = state_bridge.query_megatron(unwrapped)
                if not bridge:
                    continue

                # ensure src weights exist
                if isinstance(bridge.to, str):
                    if bridge.to not in state_accessor:
                        continue
                else:
                    if any(v not in state_accessor for v in bridge.to.values()):
                        continue

                owner_module = self._unwrap_model(model)
                for part in unwrapped.split(".")[:-1]:
                    owner_module = getattr(owner_module, part)

                yield _HFLoadTask(
                    model_idx=model_idx,
                    param_name=self._unwrap_name(name),
                    megatron_module=owner_module,
                    megatron_param=param,
                    bridge=bridge,
                )

    def _build_plan_to_hf(
        self,
        megatron_models: list[MegatronModel],
        hf_pretrained: HFPreTrained,
        order: Literal["megatron", "hf", "safetensors"],
    ) -> Iterable[_HFSaveTask]:
        """Construct the *Megatron ➜ HF* save plan.

        Parameters
        ----------
        megatron_models
            List of local Megatron *pipeline* replicas (length ≥ 1, length > 1
            only when virtual-pipeline-parallelism (VPP) is enabled).
        hf_pretrained
            HF model whose *state* object provides ordering information when
            *order* is ``'hf'`` or ``'safetensors'``.
        order
            • ``'src'`` – follow the natural order of Megatron parameters  
            • ``'hf'``  – follow the order of keys in the HF state-dict source  
            • ``'safetensors'`` – group by file name, then by key (mimics the
              original safetensors file order).
        """

        if order == "megatron":
            sb = self.state_bridge()
            for pp_rank, vpp_rank, name in self._collect_all_params(megatron_models):
                unwrapped = self._unwrap_name(name)
                bridge = sb.query_megatron(unwrapped)
                if bridge:
                    yield _HFSaveTask(pp_rank, vpp_rank, unwrapped, bridge)
            return

        # --- Otherwise: follow HF / safetensors order --------------------------------
        if not (
            hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")
        ):
            raise ValueError(
                f"order='{order}' requires hf_pretrained.state.source to be present"
            )

        if order == "hf":
            hf_keys: Iterable[str] = hf_pretrained.state.source.get_all_keys()
        elif order == "safetensors":
            if not hasattr(hf_pretrained.state.source, "key_to_filename_map"):
                raise TypeError("order='safetensors' requires the state source to have a 'key_to_filename_map'.")
            
            key_to_filename: Mapping[str, str] = hf_pretrained.state.source.key_to_filename_map
            filename_to_keys = defaultdict(list)
            for key, filename in key_to_filename.items():
                filename_to_keys[filename].append(key)

            hf_keys = (
                key
                for fname in sorted(filename_to_keys.keys())
                for key in filename_to_keys[fname]
            )
        else:
            raise ValueError(f"Invalid order: {order}, supported orders are 'megatron', 'hf' and 'safetensors'")

        sb = self.state_bridge()
        emitted = set()

        param_locations = defaultdict(list)
        for pp_rank, vpp_rank, param_name in self._collect_all_params(megatron_models):
            unwrapped_name = self._unwrap_name(param_name)
            param_locations[unwrapped_name].append((pp_rank, vpp_rank, param_name))

        for hf_key in hf_keys:
            bridge = sb.query_to(hf_key)
            if not bridge:
                if hf_key == "model.layers.0.mlp.gate_proj.weight":
                    bridge = sb.query_to(hf_key)
                continue
            
            src_name = bridge.megatron if hasattr(bridge, "megatron") else None
            if not src_name or src_name in emitted:
                continue

            if src_name not in param_locations:
                continue

            emitted.add(src_name)
            pp, vpp, _ = sorted(param_locations[src_name])[0]
            yield _HFSaveTask(pp, vpp, src_name, bridge)


def is_tensor_parallel(param) -> bool:
    """Check if a parameter is tensor parallel distributed."""
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
