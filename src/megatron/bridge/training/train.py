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

import gc
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Optional, Union

import torch
import torch.profiler
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunDataIterator, get_rerun_state_machine
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.utils import check_param_hashes_across_dp_replicas, get_model_config

from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import maybe_finalize_async_save, save_checkpoint
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.initialize import destroy_global_state
from megatron.bridge.training.nvrx_straggler import (
    check_nvrx_straggler_detection,
    safe_shutdown_nvrx_straggler_manager,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils import flop_utils
from megatron.bridge.training.utils.log_utils import append_to_progress_log, barrier_and_log
from megatron.bridge.training.utils.train_utils import (
    calc_params_l2_norm,
    check_forward_step_func_num_args,
    logical_and_across_model_parallel_group,
    maybe_inject_state,
    reduce_max_stat_across_model_parallel_group,
    training_log,
)
from megatron.bridge.utils.common_utils import get_world_size_safe, print_rank_0


def train(
    forward_step_func: ForwardStepCallable,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer,
    scheduler: OptimizerParamScheduler,
    train_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    valid_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    global_state: GlobalState,
    checkpointing_context: dict[str, Any],
    process_non_loss_data_func: Optional[Callable] = None,
    non_loss_data_func: Optional[Callable] = None,
) -> None:
    """Main training loop.

    Handles the overall training process, including the iteration loop,
    calling train_step, evaluation, checkpointing, logging, and exit conditions.

    Args:
        forward_step_func: Callable that executes a single forward step.
        model: list of model chunks (potentially wrapped in DDP).
        optimizer: The optimizer instance.
        scheduler: The learning rate scheduler instance.
        train_data_iterator: Iterator for the training dataset.
        valid_data_iterator: Iterator for the validation dataset.
        global_state: The GlobalState object holding various training states.
        checkpointing_context: Context dictionary for checkpointing.
        process_non_loss_data_func: Optional function to process non-loss data during evaluation.
        non_loss_data_func: Optional function to compute non-loss data during evaluation.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    config: ConfigContainer = global_state.cfg
    model_config = get_model_config(model[0])
    train_config = config.train
    timers = global_state.timers
    straggler_timer = global_state.straggler_timer
    energy_monitor = global_state.energy_monitor

    # Check num args to forward_step_func
    num_fw_args = check_forward_step_func_num_args(forward_step_func)

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Make sure rerun_state_machine has the right iteration loaded from checkpoint.
    rerun_state_machine = get_rerun_state_machine()
    if rerun_state_machine.current_iteration != global_state.train_state.step:
        print_rank_0(f"Setting rerun_state_machine.current_iteration to {global_state.train_state.step}...")
        rerun_state_machine.current_iteration = global_state.train_state.step

    num_floating_point_operations_so_far = global_state.train_state.floating_point_operations_so_far
    num_floating_point_operations_since_last_log_event = 0.0

    if energy_monitor is not None:
        energy_monitor.setup()
        energy_monitor.resume()

    timers("interval-time", log_level=0).start(barrier=True)
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if train_config.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert train_config.manual_gc_interval >= 0, (
            "Manual garbage collection interval should be larger than or equal to 0"
        )
        gc.disable()
        gc.collect()

    if config.straggler and config.straggler.log_straggler:
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = config.straggler.straggler_minmax_count
        straggler_timer.configure(
            world,
            rank,
            mmcnt=mmcnt,
            enabled=not config.straggler.disable_straggler_on_startup,
            port=config.straggler.straggler_ctrlr_port,
        )

    # Initialize NVRx straggler detection if enabled
    nvrx_straggler_manager = global_state.nvrx_straggler_manager
    if nvrx_straggler_manager is not None:
        try:
            # Initialize the straggler detector first
            nvrx_straggler_manager.initialize()
            # Wrap the train_step function for monitoring
            # Note: The nvidia-resiliency-ext library will monitor the actual train_step calls
            nvrx_straggler_manager.wrap_train_step_function(train_step)
        except Exception as e:
            print_rank_0(f"Failed to initialize NVRx straggler detection: {e}")
            # Set to None to disable further checks
            global_state._nvrx_straggler_manager = None

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    prof = None
    prof_config = config.profiling
    if prof_config and torch.distributed.get_rank() in prof_config.profile_ranks and prof_config.use_pytorch_profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(prof_config.profile_step_start - 1, 0),
                warmup=1 if prof_config.profile_step_start > 0 else 0,
                active=prof_config.profile_step_end - prof_config.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.logger.tensorboard_dir),
            record_shapes=prof_config.record_shapes,
            with_stack=True,
        )
        prof.start()

    start_iteration = global_state.train_state.step
    # Megatron FSDP and FSDP2 does not have this hook
    should_toggle_forward_pre_hook = should_disable_forward_pre_hook(
        config.ddp.use_megatron_fsdp,
        config.optimizer.use_distributed_optimizer,
        config.ddp.overlap_param_gather,
    )
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_toggle_forward_pre_hook:
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = model_config.param_sync_func
        model_config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if train_config.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), (
            "Parameter hashes not matching across DP replicas"
        )
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {global_state.train_state.step} iterations...")

    # Capture CUDA Graphs.
    if model_config.external_cuda_graph:
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=model_config,
            seq_length=config.model.seq_length,
            micro_batch_size=config.train.micro_batch_size,
            optimizers=[optimizer],
        )
        cuda_graph_helper.create_cudagraphs()

    # Run training iterations till done.
    while global_state.train_state.step < train_config.train_iters:
        if prof_config and torch.distributed.get_rank() in prof_config.profile_ranks:
            if prof_config.use_pytorch_profiler:
                prof.step()
            if prof_config.use_nsys_profiler:
                if global_state.train_state.step == prof_config.profile_step_start:
                    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
                    torch.autograd.profiler.emit_nvtx(record_shapes=prof_config.record_shapes).__enter__()

        fault_tolerance.on_checkpointing_start(global_state)
        maybe_finalize_async_save(global_state=global_state, ckpt_cfg=config.checkpoint, blocking=False)
        fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(global_state.train_state.consumed_train_samples, consistency_check=False, verbose=True)
        if get_num_microbatches() != num_microbatches and global_state.train_state.step != 0:
            assert get_num_microbatches() > num_microbatches, (
                f"Number of microbatches should be increasing due to batch size rampup; "
                f"instead going from {num_microbatches} to {get_num_microbatches()}"
            )
            if config.checkpoint.save is not None:
                save_checkpoint_and_time(
                    global_state,
                    model,
                    optimizer,
                    scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    non_persistent_ckpt=False,  # TODO: implement non-persistent checkpointing
                    train_data_iterator=train_data_iterator,
                )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(global_state.train_state.consumed_train_samples, consistency_check=True, verbose=True)

        # TODO: implement dummy train_step to fast forward train_data_iterator.
        # Completely skip iteration if needed.
        # if global_state.train_state.step in config.checkpoint.iterations_to_skip:
        #     # Dummy train_step to fast forward train_data_iterator.
        #     dummy_train_step(train_data_iterator)
        #     global_state.train_state.step += 1
        #     batch_size = (
        #         parallel_state.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        #     )
        #     global_state.train_state.consumed_train_samples += batch_size
        #     global_state.train_state.skipped_train_samples += batch_size
        #     continue

        # Run training step.
        fault_tolerance.on_training_step_start(global_state)
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = train_step(
            forward_step_func, num_fw_args, train_data_iterator, model, optimizer, scheduler, global_state
        )
        fault_tolerance.on_training_step_end(global_state)
        if should_checkpoint:
            save_checkpoint_and_time(
                global_state,
                model,
                optimizer,
                scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
                non_persistent_ckpt=False,  # TODO: implement non-persistent checkpointing
            )
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if global_state.train_state.step == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = global_state.train_state.step + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_toggle_forward_pre_hook:
                    enable_forward_pre_hook(model)
                    model_config.param_sync_func = param_sync_func
                    pre_hook_enabled = True
                    # Set the manual hooks when CUDA Graphs are used.
                    if model_config.external_cuda_graph:
                        cuda_graph_helper.cuda_graph_set_manual_hooks()

        global_state.train_state.step += 1
        batch_size = (
            parallel_state.get_data_parallel_world_size() * train_config.micro_batch_size * get_num_microbatches()
        )
        global_state.train_state.consumed_train_samples += batch_size
        num_skipped_samples_in_batch = get_current_global_batch_size() - get_current_running_global_batch_size()
        if train_config.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        global_state.train_state.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = flop_utils.num_floating_point_operations(config, batch_size)
        global_state.train_state.floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_so_far = global_state.train_state.floating_point_operations_so_far
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        if hasattr(optimizer, "is_stub_optimizer") and not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if config.logger.log_params_norm:
            params_norm = calc_params_l2_norm(model, model_config, use_megatron_fsdp=config.dist.use_megatron_fsdp)
        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if len(param_group) == 0:
                continue
            if param_group["is_decoupled_lr"]:
                decoupled_learning_rate = param_group["lr"]
            else:
                learning_rate = param_group["lr"]
        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            config,
            global_state,
        )

        if (
            global_state.train_state.do_valid
            and train_config.eval_interval
            and global_state.train_state.step % train_config.eval_interval == 0
        ):
            if energy_monitor is not None:
                energy_monitor.pause()
            timers("interval-time").stop()
            if should_toggle_forward_pre_hook:
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f"iteration {global_state.train_state.step}"
            timers("eval-time", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                global_state,
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                model_config,
                verbose=False,
                write_to_tensorboard=True,
                process_non_loss_data_func=process_non_loss_data_func,
                non_loss_data_func=non_loss_data_func,
            )
            eval_duration += timers("eval-time").elapsed()
            eval_iterations += train_config.eval_iters
            timers("eval-time").stop()

            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if should_toggle_forward_pre_hook:
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
            timers("interval-time", log_level=0).start(barrier=True)
            if energy_monitor is not None:
                energy_monitor.resume()

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(
            model,
            num_floating_point_operations_since_last_log_event,
            straggler_timer,
            global_state.train_state.step,
            prof,
            config,
            should_toggle_forward_pre_hook,
        )

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(
            global_state,
            model,
            optimizer,
            scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator,
        )
        if should_exit:
            break

    # Flush TensorBoard, WandB writers and one-logger.
    writer = global_state.tensorboard_logger
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    # This will finalize all unfinalized async request and terminate
    # a persistent async worker if persistent ckpt worker is enabled
    fault_tolerance.on_checkpointing_start(global_state)
    maybe_finalize_async_save(global_state=global_state, ckpt_cfg=config.checkpoint, blocking=True, terminate=True)
    fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)

    # Shutdown NVRx straggler detection if enabled
    safe_shutdown_nvrx_straggler_manager(global_state.nvrx_straggler_manager)

    if energy_monitor is not None:
        energy_monitor.lap()
        total_energy = energy_monitor.get_total()
        print_rank_0(f"Total training energy (GPU): {total_energy / 1e6} MJ")
        energy_monitor.shutdown()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        maybe_finalize_async_save(global_state=global_state, ckpt_cfg=config.checkpoint, blocking=True, terminate=True)
        wandb_writer = global_state.wandb_logger
        if wandb_writer:
            wandb_writer.finish()
        fault_tolerance.shutdown(global_state)
        sys.exit(exit_code)


def train_step(
    forward_step_func: ForwardStepCallable,
    num_fw_args: int,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    model: list[MegatronModule],
    optimizer: MegatronOptimizer,
    scheduler: OptimizerParamScheduler,
    global_state: GlobalState,
) -> tuple[dict[str, torch.Tensor], int, bool, bool, int, Optional[float], Optional[int]]:
    """Single training step.

    Args:
        forward_step_func: Function that performs a forward step
        num_fw_args: Number of arguments expected by forward_step_func
        data_iterator: Iterator over training data
        model: list of model chunks
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler
        global_state: Global training state

    Returns:
        tuple containing:
        - loss_dict: Dictionary of reduced losses
        - skipped_iter: Whether the iteration was skipped (1) or not (0)
        - should_checkpoint: Whether a checkpoint should be saved
        - should_exit: Whether training should exit
        - exit_code: Exit code if should_exit is True
        - grad_norm: Gradient norm if available, None otherwise
        - num_zeros_in_grad: Number of zeros in gradient if available, None otherwise
    """
    cfg: ConfigContainer = global_state.cfg
    timers = global_state.timers
    model_config = get_model_config(model[0])
    train_config = cfg.train
    optim_config = cfg.optimizer

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Optionally inject state into forward step
        wrapped_forward_step = maybe_inject_state(forward_step_func, global_state, num_fw_args=num_fw_args)

        _handle_mxfp8_param_buffer_copy(
            optimizer=optimizer,
            reuse_grad_buf_for_mxfp8_param_ag=cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag,
            overlap_param_gather=cfg.ddp.overlap_param_gather,
        )

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=wrapped_forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=model_config.seq_length,
            micro_batch_size=train_config.micro_batch_size,
            decoder_seq_length=model_config.seq_length,
            forward_only=False,
        )
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if train_config.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=optim_config.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers("optimizer").stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if optim_config.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * train_config.micro_batch_size * cfg.data_parallel_size
        scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if train_config.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}

        for key in losses_reduced[0].keys():
            val = [x[key].view(-1) for x in losses_reduced]
            if val[0].numel() == 2:
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                val = torch.vstack(val).sum(dim=0)
                torch.distributed.all_reduce(
                    val, group=parallel_state.get_data_parallel_group(with_context_parallel=True)
                )
                loss_reduced[key] = val[0] / val[1]
            elif val[0].numel() == 1:
                # legacy behavior, we average over the number of microbatches
                val = torch.cat(val).mean()
                loss_reduced[key] = val
            else:
                raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")
        return (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad


def post_training_step_callbacks(
    model: list[MegatronModule],
    num_floating_point_operations_since_last_log_event: float,
    straggler_timer: Any,
    iteration: int,
    prof: Optional[torch.profiler.profile],
    config: ConfigContainer,
    should_toggle_forward_pre_hook: bool,
) -> None:
    """Run all post-training-step functions (e.g., FT heartbeats, GC).

    Args:
        model: list of model chunks wrapped in DDP
        num_floating_point_operations_since_last_log_event: Number of floating point operations since last log
        straggler_timer: Timer for straggler detection
        iteration: Current training iteration
        prof: PyTorch profiler instance
        config: Configuration container
        should_toggle_forward_pre_hook: Whether to toggle forward pre-hook
    """
    train_config = config.train

    # Bring CPU and GPU back in sync if on right iteration.
    if train_config.train_sync_interval and iteration % train_config.train_sync_interval == 0:
        torch.cuda.synchronize()

    # Straggler detector.
    if config.straggler:
        if iteration % config.logger.log_interval == 0 and config.straggler.log_straggler:
            straggler_timer.report(
                num_floating_point_operations_since_last_log_event,
                config.logger.log_interval,
            )
            num_floating_point_operations_since_last_log_event = 0.0

    # Check weight hash across DP replicas.
    if (
        train_config.check_weight_hash_across_dp_replicas_interval is not None
        and iteration % train_config.check_weight_hash_across_dp_replicas_interval == 0
    ):
        if should_toggle_forward_pre_hook:
            disable_forward_pre_hook(model)
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), (
            "Parameter hashes not matching across DP replicas"
        )
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")
        if should_toggle_forward_pre_hook:
            enable_forward_pre_hook(model)

    # Profiling.
    if (
        config.profiling
        and iteration == config.profiling.profile_step_end
        and torch.distributed.get_rank() in config.profiling.profile_ranks
    ):
        if config.profiling.use_pytorch_profiler:
            assert prof is not None
            prof.stop()
        if config.profiling.use_nsys_profiler:
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())

    # Manual garbage collection.
    if train_config.manual_gc:
        if train_config.manual_gc_interval != 0 and iteration % train_config.manual_gc_interval == 0:
            gc.collect()


def should_disable_forward_pre_hook(
    use_megatron_fsdp: bool, use_distributed_optimizer: bool, overlap_param_gather: bool
) -> bool:
    """Determine if forward pre-hooks should be disabled during checkpointing.

    Forward pre-hooks need to be disabled during checkpoint saving when using
    distributed optimizer with overlapped parameter gathering

    Args:
        use_megatron_fsdp: Whether Megatron FSDP is enabled.
        use_distributed_optimizer: Whether distributed optimizer is enabled.
        overlap_param_gather: Whether parameter gathering is overlapped.

    Returns:
        True if forward pre-hooks should be disabled, False otherwise.

    Note:
        This is needed to prevent autograd issues during checkpoint saving
        when using distributed optimizer with parameter gathering overlap.
    """
    return not use_megatron_fsdp and use_distributed_optimizer and overlap_param_gather


def enable_forward_pre_hook(model: list[DDP]) -> None:
    """Enable forward pre-hook for all model chunks.

    Args:
        model: list of model chunks wrapped in DDP
    """
    for model_chunk in model:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model: list[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hook for all model chunks.

    Args:
        model: list of model chunks wrapped in DDP
        param_sync: Whether to synchronize parameters across model chunks
    """
    for model_chunk in model:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def get_start_time_from_progress_log(cfg: ConfigContainer) -> tuple[datetime, float]:
    """
    Gets start time of earliest job with same world size. Also returns the number
    of floating-point operations completed in last saved checkpoint.
    """
    assert cfg.checkpoint.save is not None
    progress_log_filename = os.path.join(cfg.checkpoint.save, "progress.txt")

    # start_time is time when job with same world size started.
    # start_num_floating_point_operations is the number of floating-point operations
    # completed when this job started.
    # latest_num_floating_point_operations is the number of floating-point operations
    # completed in most recent saved checkpoint.
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0

    def _get_field(string, type):
        return type(string.split(": ")[1])

    with open(progress_log_filename, "r") as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split("\t")
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = _get_field(line_tokens[7], float)
            if world_size_in_line != get_world_size_safe():
                # Re-start search if we see a different world size.
                start_time = None
                start_num_floating_point_operations = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = latest_num_floating_point_operations
    assert start_time is not None and start_num_floating_point_operations is not None, (
        "Should have seen at least one 'Starting job' entry with same world_size"
    )
    return datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"), start_num_floating_point_operations


def compute_throughputs_and_append_to_progress_log(
    state: GlobalState, num_floating_point_operations_so_far: float
) -> None:
    """Computes job and cumulative throughputs and appends to progress log.

    Calculates TFLOP/s/GPU based on floating-point operations and elapsed time.
    Appends the computed throughputs, total FLOPs, and processed tokens to the
    progress log file.

    Args:
        state: The GlobalState object.
        num_floating_point_operations_so_far: Total floating-point operations completed.
    """
    if state.cfg.checkpoint.save is None:
        return

    # Compute job throughput.
    # args.num_floating_point_operations_so_far keeps track of floating-point operations
    # completed at the start of job.
    job_throughput = (num_floating_point_operations_so_far - state.train_state.floating_point_operations_so_far) / (
        (time.time() - state.start_time) * 10**12 * get_world_size_safe()
    )

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log(state.cfg)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = (num_floating_point_operations_so_far - start_num_floating_point_operations) / (
        elapsed_time * 10**12 * get_world_size_safe()
    )

    tokens_so_far = state.train_state.consumed_train_samples * state.cfg.model.seq_length
    saved_ckpt_prefix = "Saving async checkpoint" if state.cfg.checkpoint.async_save else "Saved checkpoint"
    append_to_progress_log(
        state.cfg.checkpoint.save,
        f"{saved_ckpt_prefix}\tIteration: {state.train_state.step}\t"
        f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
        f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
        f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
        f"Tokens (in billions): {tokens_so_far / 10**9:.2f}",
    )


def save_checkpoint_and_time(
    state: GlobalState,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_floating_point_operations_so_far: float,
    checkpointing_context: dict[str, Any],
    non_persistent_ckpt: bool = False,
    train_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]] = None,
) -> None:
    """Saves a checkpoint and logs the timing.

    Wraps the `save_checkpoint` function with timers and potentially disables/
    enables forward pre-hooks if distributed optimizer with overlapped parameter
    gather is used.

    Args:
        state: The global state object.
        model: list of model chunks (MegatronModule instances).
        optimizer: The optimizer instance.
        opt_param_scheduler: The optimizer parameter scheduler instance.
        num_floating_point_operations_so_far: Cumulative TFLOPs up to this point.
        checkpointing_context: Dictionary holding checkpointing-related state.
        non_persistent_ckpt: Flag indicating if this is a non-persistent
                             (local) checkpoint. Defaults to False.
        train_data_iterator: Optional training data iterator to save its state.
    """
    timers = state.timers
    energy_monitor = state.energy_monitor

    # Stop timer to get accurate train interval time and exclude checkpointing duration
    timers("interval-time").stop()

    # Pause energy monitor
    if energy_monitor is not None:
        energy_monitor.pause()

    # Extra barrier is added to make sure all ranks report the max time.
    timer_key = "save-checkpoint-non-persistent" if non_persistent_ckpt else "save-checkpoint"
    timers(timer_key, log_level=0).start(barrier=True)

    should_disable_pre_hook = should_disable_forward_pre_hook(
        state.cfg.ddp.use_megatron_fsdp,
        state.cfg.optimizer.use_distributed_optimizer,
        state.cfg.ddp.overlap_param_gather,
    )
    if should_disable_pre_hook:
        disable_forward_pre_hook(model)
    save_checkpoint(
        state,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        checkpointing_context=checkpointing_context,
        non_persistent_ckpt=non_persistent_ckpt,
        train_data_iterator=train_data_iterator,
    )
    if state.cfg.model.fp8 is not None:
        # Run garbage collection after checkpoint saving to free memory from
        # dequantized bf16 tensors that were temporarily created during fp8
        # model checkpoint saving.
        gc.collect()
    if should_disable_pre_hook:
        enable_forward_pre_hook(model)
    timers(timer_key).stop(barrier=True)
    timers.log([timer_key])

    if state.cfg.logger.log_progress and not non_persistent_ckpt:
        compute_throughputs_and_append_to_progress_log(state, num_floating_point_operations_so_far)

    # Recover timing
    if energy_monitor is not None:
        energy_monitor.resume()
    timers("interval-time", log_level=0).start(barrier=True)


def checkpoint_and_decide_exit(
    state: GlobalState,
    model: list[MegatronModule],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_floating_point_operations_so_far: float,
    checkpointing_context: dict[str, Any],
    train_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
) -> bool:
    """Handles checkpointing decisions and determines if training should exit.

    Checks various conditions for saving a checkpoint (signal received, interval,
    duration) and determines if the training loop should terminate based on exit
    conditions (signal, duration, iteration interval).

    Args:
        state: The global state object.
        model: list of model chunks (MegatronModule instances).
        optimizer: The optimizer instance.
        opt_param_scheduler: The optimizer parameter scheduler instance.
        num_floating_point_operations_so_far: Cumulative TFLOPs up to this point.
        checkpointing_context: Dictionary holding checkpointing-related state.
        train_data_iterator: Optional training data iterator to save its state.

    Returns:
        True if the training loop should exit, False otherwise.
    """
    saved_checkpoint = False

    # Exit based on signal handler.
    if state.cfg.train.exit_signal_handler:
        signal_handler = state.signal_handler
        if any(signal_handler.signals_received()):
            if state.cfg.checkpoint.save:
                save_checkpoint_and_time(
                    state,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator=train_data_iterator,
                )
            barrier_and_log("exiting program after receiving SIGTERM.")

            return True

    # Regular save (persistent and non-persistent).
    if (
        state.cfg.checkpoint.save
        and state.cfg.checkpoint.save_interval
        and state.train_state.step % state.cfg.checkpoint.save_interval == 0
    ):
        save_checkpoint_and_time(
            state,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator=train_data_iterator,
        )
        saved_checkpoint = True

    elif (
        state.cfg.checkpoint.save
        and state.cfg.checkpoint.non_persistent_save_interval
        and state.train_state.step % state.cfg.checkpoint.non_persistent_save_interval == 0
    ):
        save_checkpoint_and_time(
            state,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            non_persistent_ckpt=True,
            train_data_iterator=train_data_iterator,
        )
        saved_checkpoint = True

    # Exit based on duration.
    if state.cfg.train.exit_duration_in_mins:
        train_time = (time.time() - state.start_time) / 60.0
        done_cuda = torch.tensor([train_time > state.cfg.train.exit_duration_in_mins], dtype=torch.int, device="cuda")
        torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
        done = done_cuda.item()
        if done:
            if state.cfg.checkpoint.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    state,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator=train_data_iterator,
                )
            barrier_and_log(f"exiting program after {train_time} minutes")

            return True

    # Exit based on iterations.
    if state.cfg.train.exit_interval and state.train_state.step % state.cfg.train.exit_interval == 0:
        if state.cfg.checkpoint.save and not saved_checkpoint:
            save_checkpoint_and_time(
                state,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
            )
        barrier_and_log(f"exiting program at iteration {state.train_state.step}")

        return True

    # Exit based on NVRx straggler detection
    if check_nvrx_straggler_detection(state.nvrx_straggler_manager):
        if state.cfg.checkpoint.save is not None and not saved_checkpoint:
            save_checkpoint_and_time(
                state,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
            )
        barrier_and_log("Exiting program due to straggler detection.")
        return True

    return False


def _finish_train(global_state: GlobalState):
    ckpt_cfg = global_state.cfg.checkpoint

    # Shutdown NVRx straggler detection if enabled
    safe_shutdown_nvrx_straggler_manager(global_state.nvrx_straggler_manager)

    fault_tolerance.on_checkpointing_start(global_state)
    maybe_finalize_async_save(global_state=global_state, blocking=True, terminate=True, ckpt_cfg=ckpt_cfg)
    fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)
    fault_tolerance.shutdown(global_state)

    if global_state.wandb_logger:
        global_state.wandb_logger.finish()

    destroy_global_state()


def _handle_mxfp8_param_buffer_copy(
    optimizer: MegatronOptimizer, reuse_grad_buf_for_mxfp8_param_ag: bool, overlap_param_gather: bool
) -> None:
    """Copy main params to param buffer for mxfp8 with grad buffer reuse.

    For mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
    we need to call _copy_main_params_to_param_buffer() after the grad buffer
    is zeroed because param and grad buffer are shared.

    Args:
        optimizer: The MegatronOptimizer instance
        reuse_grad_buf_for_mxfp8_param_ag: Config flag for grad buffer reuse
        overlap_param_gather: Config flag for overlapping param gathering
    """
    if reuse_grad_buf_for_mxfp8_param_ag and overlap_param_gather:
        for optim_instance in optimizer.chained_optimizers:
            if isinstance(optim_instance, DistributedOptimizer):
                optim_instance._copy_main_params_to_param_buffer()
