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

import inspect
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.theoretical_memory_utils import report_theoretical_memory
from megatron.bridge.utils.common_utils import get_world_size_safe, is_last_rank, print_rank_last


try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        import warnings

        warnings.warn(
            "Transformer Engine and Apex are not installed. "
            "Falling back to local implementations of "
            "multi_tensor_applier and multi_tensor_l2norm"
        )

        from megatron.core.utils import local_multi_tensor_applier as multi_tensor_applier
        from megatron.core.utils import local_multi_tensor_l2_norm as multi_tensor_l2norm


def param_is_not_shared(param: nn.Parameter) -> bool:
    """Check if a parameter is marked as not shared.

    Args:
        param (torch.nn.Parameter): The parameter to check.

    Returns:
        bool: True if the parameter does not have a 'shared' attribute or if
              param.shared is False.
    """
    return not hasattr(param, "shared") or not param.shared


def calc_params_l2_norm(
    model: Union[MegatronModule, list[MegatronModule]],
    model_config: Any,
    use_megatron_fsdp: bool = False,
    force_create_fp32_copy: bool = False,
) -> float:
    """Calculate the L2 norm of model parameters across all GPUs.

    Handles parameter sharding (DP, TP, PP, EP) and different parameter types
    (dense, MoE, sharded main params).

    Args:
        model (Union[torch.nn.Module, list[torch.nn.Module]]): The model or list of model chunks.
        model_config: The model configuration object.
        force_create_fp32_copy (bool, optional): If True, always creates an FP32 copy
            for norm calculation, ignoring potential `main_param` attributes.
            Defaults to False.

    Returns:
        float: The L2 norm of all parameters.
    """
    if not isinstance(model, list):
        model = [model]

    if use_megatron_fsdp:
        # All Megatron FSDP parameters are expected to be PyTorch DTensor.
        # params_data is a dict of device_mesh -> list of local tensors.
        params = []
        for model_chunk in model:
            model_chunk.stop_communication()
            for name, param in model_chunk.named_parameters():
                if not hasattr(param, "_local_tensor"):
                    raise RuntimeError(
                        f"Megatron FSDP requires parameters are PyTorch DTensor. Parameter {name} is not a DTensor."
                    )
                params.append(param)

        return calc_dtensor_params_l2_norm(params)

    # Seperate moe and dense params
    params_data = []
    moe_params_data = []
    sharded_params_data = []
    data_parallel_group = None

    for model_chunk in model:
        for param in model_chunk.parameters():
            data_parallel_group = get_data_parallel_group_if_dtensor(param, data_parallel_group)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if not is_not_tp_duplicate:
                continue
            assert is_not_tp_duplicate
            if not getattr(param, "allreduce", True):
                # TODO: Implement memory optimization for MoE parameters.
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                moe_params_data.append(param.data.float() if model_config.bf16 else param.data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    if model_config.bf16:
                        if not force_create_fp32_copy and hasattr(param, "main_param"):
                            if getattr(param, "main_param_sharded", False):
                                if param.main_param is not None:
                                    sharded_params_data.append(param.main_param)
                            else:
                                params_data.append(param.main_param)
                        else:
                            # Fallback to original logic of making a fp32 copy of the
                            # parameter if `.main_param` attribute is not available.
                            params_data.append(param.data.float())
                    else:
                        params_data.append(param.data)

    # Calculate norm.
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
    if len(params_data) > 0:
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [params_data],
            False,  # no per-parameter norm.
        )
        norm_2 = norm * norm
    else:
        norm_2 = torch.zeros((1,), dtype=torch.float32, device="cuda")

    if data_parallel_group is not None:
        torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group)

    # Add norm contribution from params with sharded main_params. These norms need to be
    # accumulated across the DP group since the main parameters are sharded because
    # of distributed optimizer.
    if len(sharded_params_data) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        sharded_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [sharded_params_data],
            False,  # no per-parameter norm.
        )
        sharded_norm_2 = sharded_norm * sharded_norm
        # Sum over all DP groups, including CP since distributed optimizer state is
        # sharded jointly over DP+CP.
        torch.distributed.all_reduce(
            sharded_norm_2,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )
        norm_2 += sharded_norm_2

    # Add norm contribution from expert layers in MoEs.
    if len(moe_params_data) > 0:
        moe_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [moe_params_data],
            False,  # no per-parameter norm.
        )
        moe_norm_2 = moe_norm * moe_norm

    # Account for MoE norm even if current rank doesn't have any expert params to prevent
    # hang in models with un-even numbers of MoE layers.
    # See details in https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/409
    else:
        moe_norm_2 = torch.zeros_like(norm_2)

    # Reduce norm across model parallel groups (dense and expert).
    # Dense params should sum across all model-parallel GPUs (tensor + pipeline).
    dense_reduce_group = parallel_state.get_model_parallel_group()
    ranks_in_dense_reduce_group = torch.distributed.get_process_group_ranks(dense_reduce_group)
    # Expert params should sum across all model-parallel GPUs (expert + tensor + pipeline).
    expert_reduce_group = parallel_state.get_expert_tensor_model_pipeline_parallel_group()
    ranks_in_expert_reduce_group = torch.distributed.get_process_group_ranks(expert_reduce_group)

    # If dense and expert reduce groups are the same, sum then reduce.
    if ranks_in_dense_reduce_group == ranks_in_expert_reduce_group:
        norm_2 += moe_norm_2
        torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group)
    # If dense and expert reduce groups are different, reduce then sum.
    else:
        torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group)
        torch.distributed.all_reduce(moe_norm_2, op=torch.distributed.ReduceOp.SUM, group=expert_reduce_group)
        norm_2 += moe_norm_2

    return norm_2.item() ** 0.5


def calc_dtensor_params_l2_norm(params):
    """Calculate l2 norm of DTensor parameters."""
    params_data = defaultdict(list)
    for param in params:
        params_data[param._spec].append(param._local_tensor)

    total_norm_2 = torch.zeros((1,), dtype=torch.float32, device="cuda")
    dummy_overflow_buf = torch.zeros((1,), dtype=torch.int, device="cuda")
    for dtensor_spec, local_tensors in params_data.items():
        local_tensors = [t for t in local_tensors if t.numel() > 0]
        if len(local_tensors) == 0:
            norm = torch.zeros((1,), dtype=torch.float32, device="cuda")
        else:
            norm, _ = multi_tensor_applier(
                multi_tensor_l2norm,
                dummy_overflow_buf,
                [local_tensors],
                False,  # no per-parameter norm.
            )
        norm_2 = norm * norm
        for pg, placement in zip(
            dtensor_spec.device_mesh.get_all_groups(),
            dtensor_spec.placements,
        ):
            if placement.is_shard():
                torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=pg)
            elif placement.is_replicate():
                # Replicated parameters are already summed across all ranks.
                pass
            else:
                raise RuntimeError(f"Unsupported placement {placement} for Megatron FSDP.")
        total_norm_2 += norm_2

    return total_norm_2.item() ** 0.5


def reduce_max_stat_across_model_parallel_group(stat: Optional[float]) -> Optional[float]:
    """Calculates the max of a stat across the model parallel group.

    Handles cases where some ranks might have the stat as None (e.g., grad norm
    on ranks without an optimizer).

    Args:
        stat (float): The statistic value (or None) on the current rank.

    Returns:
        float: The maximum value of the statistic across the model parallel group,
               or None if all ranks had None.
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    else:
        return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """Performs a logical AND operation across the model parallel group.

    Args:
        input (bool): The boolean value on the current rank.

    Returns:
        bool: The result of the logical AND across all ranks in the group.
    """
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=parallel_state.get_model_parallel_group()
    )
    return bool(input.item())


def training_log(
    loss_dict: dict[str, torch.Tensor],
    total_loss_dict: dict[str, Any],
    learning_rate: Optional[float],
    decoupled_learning_rate: Optional[float],
    loss_scale: float,
    report_memory_flag: bool,
    skipped_iter: int,
    grad_norm: Optional[float],
    params_norm: Optional[float],
    num_zeros_in_grad: Optional[int],
    config: ConfigContainer,
    global_state: GlobalState,
) -> bool:
    """Log training stats (losses, learning rate, timings, etc.).

    Aggregates losses, logs metrics to TensorBoard and WandB (if enabled),
    and prints a formatted log string to the console on the last rank.

    Args:
        loss_dict (dict[str, torch.Tensor]): Dictionary of losses for the current step.
        total_loss_dict (dict[str, Any]): Dictionary to accumulate losses and stats
                                         across logging intervals.
        learning_rate (Optional[float]): Current learning rate.
        decoupled_learning_rate (Optional[float]): Current decoupled learning rate (if used).
        loss_scale (float): Current loss scale value.
        report_memory_flag (bool): Flag to indicate if memory usage should be reported.
        skipped_iter (int): 1 if the iteration was skipped, 0 otherwise.
        grad_norm (Optional[float]): Gradient norm if computed, else None.
        params_norm (Optional[float]): Parameter L2 norm if computed, else None.
        num_zeros_in_grad (Optional[int]): Number of zeros in gradient if computed, else None.
        config: The main configuration container.
        global_state: The global training state.

    Returns:
        bool: The updated report_memory_flag.
    """
    timers = global_state.timers
    train_state = global_state.train_state
    iteration = train_state.step
    writer = global_state.tensorboard_logger
    wandb_writer = global_state.wandb_logger
    energy_monitor = global_state.energy_monitor
    logger_config = config.logger
    train_config = config.train

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float, device="cuda")) + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "all-grads-sync",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

    # Calculate batch size.
    batch_size = train_config.micro_batch_size * config.data_parallel_size * get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
    # Tensorboard values.
    # Timer requires all the ranks to call.
    if logger_config.log_timers_to_tensorboard and (iteration % logger_config.tensorboard_log_interval == 0):
        reset_in_tb = False if hasattr(timers, "write_to_wandb") else True
        timers.write(timers_to_log, writer, iteration, normalizer=total_iterations, reset=reset_in_tb)
        if hasattr(timers, "write_to_wandb"):
            timers.write_to_wandb(timers_to_log, wandb_writer, iteration, normalizer=total_iterations, reset=True)

    if writer and (iteration % logger_config.tensorboard_log_interval == 0):
        if config.profiling:
            if config.profiling.record_memory_history and is_last_rank():
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump

                with open(config.profiling.memory_snapshot_path, "wb") as f:
                    dump(snapshot, f)

        if wandb_writer:
            wandb_writer.log({"samples vs steps": global_state.train_state.consumed_train_samples}, iteration)
        writer.add_scalar("learning-rate", learning_rate, iteration)
        writer.add_scalar("learning-rate vs samples", learning_rate, global_state.train_state.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({"learning-rate": learning_rate}, iteration)
        if config.optimizer.decoupled_lr is not None:
            writer.add_scalar("decoupled-learning-rate", decoupled_learning_rate, iteration)
        if global_state.train_state.skipped_train_samples > 0:
            writer.add_scalar("skipped-train-samples", global_state.train_state.skipped_train_samples, iteration)
            if wandb_writer:
                wandb_writer.log({"skipped-train-samples": global_state.train_state.skipped_train_samples}, iteration)
        writer.add_scalar("batch-size", batch_size, iteration)
        writer.add_scalar("batch-size vs samples", batch_size, global_state.train_state.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({"batch-size": batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(key + " vs samples", loss_dict[key], global_state.train_state.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if logger_config.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale", loss_scale, iteration)
            writer.add_scalar("loss-scale vs samples", loss_scale, global_state.train_state.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"loss-scale": loss_scale}, iteration)
        if logger_config.log_world_size_to_tensorboard:
            writer.add_scalar("world-size", get_world_size_safe(), iteration)
            writer.add_scalar(
                "world-size vs samples", get_world_size_safe(), global_state.train_state.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"world-size": get_world_size_safe()}, iteration)
        if grad_norm is not None:
            writer.add_scalar("grad-norm", grad_norm, iteration)
            writer.add_scalar("grad-norm vs samples", grad_norm, global_state.train_state.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"grad-norm": grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar(
                "num-zeros vs samples", num_zeros_in_grad, global_state.train_state.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"num-zeros": num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar("params-norm", params_norm, iteration)
            writer.add_scalar("params-norm vs samples", params_norm, global_state.train_state.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"params-norm": params_norm}, iteration)
        if logger_config.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-max-allocated-bytes",
                mem_stats["allocated_bytes.all.peak"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if config.model.num_moe_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_names = []
        if config.model.moe_router_load_balancing_type in ("aux_loss", "seq_aux_loss"):
            track_names.append("load_balancing_loss")
        if config.model.moe_z_loss_coeff is not None:
            track_names.append("z_loss")
        track_moe_metrics(
            loss_scale=moe_loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
            per_layer_logging=config.model.moe_per_layer_logging,
            force_initialize=True,
            track_names=track_names,
            num_layers=config.model.num_layers,
            moe_layer_freq=config.model.moe_layer_freq,
            mtp_num_layers=config.model.mtp_num_layers,
        )
    if config.model.mtp_num_layers is not None:
        mtp_loss_scale = 1 / get_num_microbatches()
        MTPLossLoggingHelper.track_mtp_metrics(mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict)

    if iteration % logger_config.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        # throughput = num_floating_point_operations(args, batch_size) / (
        #     elapsed_time_per_iteration * 10**12 * get_world_size_safe())  # TODO: implement

        if logger_config.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar("iteration-time", elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({"iteration-time": elapsed_time_per_iteration}, iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += " iteration {:8d}/{:8d} |".format(iteration, train_config.train_iters)
        log_string += " consumed samples: {:12d} |".format(global_state.train_state.consumed_train_samples)
        if global_state.train_state.skipped_train_samples > 0:
            log_string += " skipped samples: {:12d} |".format(global_state.train_state.skipped_train_samples)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(elapsed_time_per_iteration * 1000.0)

        # TODO: enable after flops is implemented
        # if logger_config.log_throughput:
        #     log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
        #     if logger_config.log_timers_to_tensorboard:
        #         if writer:
        #             writer.add_scalar('throughput', throughput, iteration)
        #         if wandb_writer:
        #             wandb_writer.log({'throughput': throughput}, iteration)

        if energy_monitor is not None:
            energy = (energy_monitor.lap() / total_iterations) / get_world_size_safe()
            power = energy / elapsed_time_per_iteration
            log_string += f" energy per GPU (J/iter/GPU): {energy:.1f} |"
            log_string += f" power per GPU (W/GPU): {power:.1f} |"
            if writer:
                writer.add_scalar("iter-energy/gpu", energy, iteration)
                writer.add_scalar("power/gpu", power, iteration)
            if wandb_writer:
                wandb_writer.log({"iter-energy/gpu": energy}, iteration)
                wandb_writer.log({"power/gpu": power}, iteration)

        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += f" learning rate: {learning_rate:.6E} |"
        if config.optimizer.decoupled_lr is not None and (
            parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
        ):
            assert decoupled_learning_rate is not None
            log_string += f" decoupled learning rate: {decoupled_learning_rate:.6E} |"
        else:
            assert decoupled_learning_rate is None
        log_string += f" global batch size: {batch_size:5d} |"
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device="cuda")
        log_string += f" loss scale: {loss_scale:.1f} |"
        if grad_norm is not None:
            log_string += f" grad norm: {grad_norm:.3f} |"
        if num_zeros_in_grad is not None:
            log_string += f" num zeros: {num_zeros_in_grad} |"
        if params_norm is not None:
            log_string += f" params norm: {params_norm:.3f} |"
        log_string += " number of skipped iterations: {:3d} |".format(total_loss_dict[skipped_iters_key])
        log_string += " number of nan iterations: {:3d} |".format(total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(config, num_microbatches=num_microbatches, verbose=True)
            report_memory(f"(after {iteration} iterations)")
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=logger_config.log_interval)

    return report_memory_flag


def report_memory(name: str) -> None:
    """Report current and peak GPU memory usage for the current rank.

    Args:
        name (str): A name to include in the output message (e.g., stage of training).
    """
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if parallel_state.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def needs_global_state_injection(forward_step_func: ForwardStepCallable) -> bool:
    """Check if a forward step function needs GlobalState injection.

    This function does the signature inspection once to determine if state should be injected.
    It's more efficient than repeated signature inspection in the training loop.

    Detection logic:
    1. First checks for GlobalState type annotation in any parameter
    2. Falls back to checking if first parameter is named 'state' or 'global_state'

    Args:
        forward_step_func: The forward step function to inspect.

    Returns:
        True if GlobalState should be injected, False otherwise.
    """
    signature = inspect.signature(forward_step_func)
    parameters = signature.parameters
    param_names = list(parameters.keys())

    # Check for GlobalState type annotation in any parameter
    for param_name, param in parameters.items():
        if param.annotation != inspect.Parameter.empty:
            # Handle both direct GlobalState and string annotations
            if (
                param.annotation == GlobalState
                or (isinstance(param.annotation, str) and "GlobalState" in param.annotation)
                or (hasattr(param.annotation, "__name__") and param.annotation.__name__ == "GlobalState")
            ):
                # Found GlobalState annotation - needs injection
                return True

    # Fallback: Check if the first parameter is named 'state' or 'global_state'
    return param_names and param_names[0] in ("state", "global_state")


def maybe_inject_state(
    forward_step_func: ForwardStepCallable, state: GlobalState, needs_injection: Optional[bool] = None
) -> ForwardStepCallable:
    """Optionally inject GlobalState into forward_step functions that expect it.

    Determines whether to inject state by inspecting function signature:
    1. First checks for GlobalState type annotation in any parameter
    2. Falls back to checking if first parameter is named 'state'
    3. Otherwise assumes the function doesn't expect state

    Supported signatures:
    - (data_iterator, model) → no injection
    - (data_iterator, model, return_schedule_plan) → no injection
    - (state: GlobalState, data_iterator, model) → inject state
    - (state: GlobalState, data_iterator, model, return_schedule_plan) → inject state
    - (state, data_iterator, model) → inject state (fallback to name-based detection)

    Args:
        forward_step_func: The original forward step function.
        state: The GlobalState object to potentially inject.
        needs_injection: Whether injection is needed (optional, will be inspected if None).
                        Pass this to avoid repeated signature inspection in training loops.

    Returns:
        The original function or a partial function with GlobalState injected.
    """
    if needs_injection is None:
        needs_injection = needs_global_state_injection(forward_step_func)

    if needs_injection:
        return partial(forward_step_func, state)
    else:
        return forward_step_func
