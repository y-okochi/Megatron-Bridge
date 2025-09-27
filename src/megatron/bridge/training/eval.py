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

import math
import time
from typing import Any, Callable, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode, get_rerun_state_machine
from megatron.core.transformer import MegatronModule

from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.train_utils import maybe_inject_state, needs_global_state_injection
from megatron.bridge.utils.common_utils import is_last_rank, print_rank_0, print_rank_last


def evaluate(
    state: GlobalState,
    forward_step_func: ForwardStepCallable,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    model: list[MegatronModule],
    process_non_loss_data_func: Optional[Callable],
    config: ConfigContainer,
    verbose: bool = False,
    non_loss_data_func: Optional[Callable] = None,
) -> tuple[Optional[dict[str, torch.Tensor]], Optional[Any], bool]:
    """Evaluation function.

    Args:
        state (GlobalState): The global state object.
        forward_step_func (Callable): The function that performs a forward step.
        data_iterator (Optional[Union[RerunDataIterator, list[RerunDataIterator]]]): Iterator over evaluation data.
        model (list[MegatronModule]): list of model chunks.
        process_non_loss_data_func (Optional[Callable]): Function to process non-loss data.
        config (ConfigContainer): Configuration container (potentially redundant).
        verbose (bool, optional): Whether to print evaluation progress. Defaults to False.
        non_loss_data_func (Optional[Callable], optional): Function to compute non-loss data. Defaults to None.

    Returns:
        tuple[Optional[dict[str, torch.Tensor]], Optional[Any], bool]: A tuple containing:
            - total_loss_dict: Dictionary of averaged losses.
            - collected_non_loss_data: Data collected by non_loss_data_func.
            - timelimit_hit: Boolean indicating if the time limit was reached.
    """
    # Check if forward_step_func needs state injection
    needs_injection = needs_global_state_injection(forward_step_func)

    timers = state.timers
    timers("evaluate", log_level=0).start(barrier=True)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = state.cfg.train.global_batch_size
    eval_num_microbatches = eval_batch_size // (state.cfg.train.micro_batch_size * state.cfg.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f"Evaluating on {state.cfg.train.eval_iters * eval_batch_size} samples")
        while iteration < state.cfg.train.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f"Evaluating iter {iteration}/{state.cfg.train.eval_iters}")

            wrapped_forward_step = maybe_inject_state(forward_step_func, state, needs_injection=needs_injection)
            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            fault_tolerance.on_eval_step_start(state)
            loss_dicts = forward_backward_func(
                forward_step_func=wrapped_forward_step,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=state.cfg.model.seq_length,
                micro_batch_size=state.cfg.train.micro_batch_size,
                forward_only=True,
            )
            fault_tolerance.on_eval_step_end(state)
            config.timers = state.timers

            # Empty unused memory
            if state.cfg.train.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for key in loss_dicts[0].keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                    val = [x[key].view(-1) for x in loss_dicts]

                    if val[0].numel() == 2:
                        val = torch.vstack(val).sum(dim=0)
                        torch.distributed.all_reduce(
                            val, group=parallel_state.get_data_parallel_group(with_context_parallel=True)
                        )
                        total_loss_dict[key] += val
                    elif val[0].numel() == 1:
                        val = torch.cat(val).sum()
                        total_loss_dict[key][0] += val
                        total_loss_dict[key][1] += len(loss_dicts)
                    else:
                        raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")

            state.train_state.consumed_valid_samples += eval_batch_size

            if state.cfg.train.exit_duration_in_mins:
                train_time = (time.time() - state.start_time) / 60.0
                done_cuda = torch.tensor(
                    [train_time > state.cfg.train.exit_duration_in_mins], dtype=torch.int, device="cuda"
                )
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    print_rank_0("Exiting during evaluation, timelimit reached")
                    return None, None, True

        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=wrapped_forward_step,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=state.cfg.model.seq_length,
                micro_batch_size=state.cfg.train.micro_batch_size,
                forward_only=True,
                collect_non_loss_data=True,
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = numerator / denominator

    timers("evaluate").stop()
    timers.log(["evaluate"])

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False


def evaluate_and_print_results(
    state: GlobalState,
    prefix: str,
    forward_step_func: ForwardStepCallable,
    data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
    model: list[MegatronModule],
    config: ConfigContainer,
    verbose: bool = False,
    write_to_tensorboard: bool = True,
    process_non_loss_data_func: Optional[Callable] = None,
    non_loss_data_func: Optional[Callable] = None,
) -> None:
    """Helper function to evaluate and dump results on screen.

    Args:
        state (GlobalState): The global state object.
        prefix (str): Prefix for logging evaluation results.
        forward_step_func (Callable): The function that performs a forward step.
        data_iterator (Optional[Union[RerunDataIterator, list[RerunDataIterator]]]): Iterator over evaluation data.
        model (list[MegatronModule]): list of model chunks.
        config (ConfigContainer): Configuration container (potentially redundant).
        verbose (bool, optional): Whether to print evaluation progress. Defaults to False.
        write_to_tensorboard (bool, optional): Whether to write results to TensorBoard. Defaults to True.
        process_non_loss_data_func (Optional[Callable], optional): Function to process non-loss data. Defaults to None.
        non_loss_data_func (Optional[Callable], optional): Function to compute non-loss data. Defaults to None.
    """
    if write_to_tensorboard:
        writer = state.tensorboard_logger
    else:
        writer = None

    wandb_writer = state.wandb_logger

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        state, forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose, non_loss_data_func
    )

    # Timelimit hit during evaluation
    if timelimit:
        return
    string = f" validation loss at {prefix} | "
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar("{} validation".format(key), total_loss_dict[key].item(), state.train_state.step)
            writer.add_scalar(
                "{} validation vs samples".format(key),
                total_loss_dict[key].item(),
                state.train_state.consumed_train_samples,
            )
            if state.cfg.logger.log_validation_ppl_to_tensorboard:
                writer.add_scalar("{} validation ppl".format(key), ppl, state.train_state.step)
                writer.add_scalar(
                    "{} validation ppl vs samples".format(key), ppl, state.train_state.consumed_train_samples
                )

        if wandb_writer and is_last_rank():
            wandb_writer.log({"{} validation".format(key): total_loss_dict[key].item()}, state.train_state.step)
            if state.cfg.logger.log_validation_ppl_to_tensorboard:
                wandb_writer.log({"{} validation ppl".format(key): ppl}, state.train_state.step)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, state.train_state.step, writer)

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)
