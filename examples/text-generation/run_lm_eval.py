# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

###############################################################################
# Copyright (C) 2020-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import json
import logging
import multiprocessing as mp
import os
import time

import lm_eval.evaluator
import lm_eval.tasks
import psutil
import torch
import torch.nn.functional as F

# Local imports
from run_generation import setup_parser
from utils import finalize_quantization, initialize_model, save_model

from optimum.habana.utils import get_hpu_memory_stats


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    pool_size = physical_cpu_count
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if world_size == 0:
        world_size = 1
    pool_size //= world_size
    if (pool_size * world_size) != physical_cpu_count:
        pool_size -= 1
    return spawn_context.Pool(pool_size)


mp.Pool = LimitedSpawnPool


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 384],
    )

    parser.add_argument(
        "--output_file", "-o", type=str, help="Output file with end results and runtime parameters", required=True
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to run",
        default=["hellaswag", "lambada_openai", "piqa", "winogrande"],
    )
    parser.add_argument("--limit_iters", type=int, help="limit examples to run that many iterations", default=None)
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument("--max_graphs", type=int, help="Maximum number of HPU graphs", default=None)
    args = setup_parser(parser)

    return args


class HabanaModelAdapter(lm_eval.base.BaseLM):
    def __init__(self, tokenizer, model, args, options):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self._batch_size = args.batch_size
        self.buckets = sorted(args.buckets)
        self.options = options
        self._device = args.device
        self.model_inputs = {"use_cache": self.options.use_cache}
        if self.model.config.model_type in [
            "llama",
            "mistral",
            "falcon",
            "phi",
            "mixtral",
            "qwen2",
            "gptj",
            "starcoder2",
            "gemma",
            "baichuan",
        ]:
            self.model_inputs.update(
                {
                    "reuse_cache": self.options.reuse_cache,
                }
            )

        if self.model.config.model_type in [
            "llama",
            "mistral",
            "qwen2",
            "falcon",
            "starcoder2",
            "gemma",
            "baichuan",
            "gpt_bigcode",
        ]:
            if self.model.config.model_type not in ["falcon", "gpt_bigcode"]:
                self.model_inputs.update(
                    {
                        "attn_softmax_bf16": self.options.attn_softmax_bf16,
                    }
                )
            self.model_inputs.update(
                {
                    "use_flash_attention": self.options.use_flash_attention,
                    "flash_attention_recompute": self.options.flash_attention_recompute,
                    "flash_attention_causal_mask": self.options.flash_attention_causal_mask,
                }
            )
            if self.model.config.model_type in ["llama", "qwen2", "baichuan", "gpt_bigcode"]:
                self.model_inputs.update({"flash_attention_fast_softmax": self.options.flash_attention_fast_softmax})
        if args.warmup:
            self.warm_up()

    def warm_up(self):
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self._batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)
            pass

    @property
    def eot_token_id(self):
        return self.model.config.eos_token_id

    @property
    def max_length(self):
        return self.buckets[-1]

    @property
    def max_gen_toks(self):
        raise NotImplementedError()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError()

    def find_bucket(self, length):
        return [b for b in self.buckets if b >= length][0]

    def _model_call(self, inps):
        bs, seq_length = inps.shape
        padding_length = 0
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.options.use_cache and self.options.reuse_cache:
                self.model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self.model.config.pad_token_id)
        logits = self.model(inps.to(self._device), **self.model_inputs)["logits"].cpu()

        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)
        return logits

    def get_model_info(self) -> dict:
        """
        Patched method to get Hugging Face model information for experiment reproducibility.
        source: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/huggingface.py/#L1375
        Remove from SynapseAI 1.21
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            elif hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            elif hasattr(model, "parameters"):
                return next(model.parameters()).dtype
            else:
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
        }
        return model_info


def main():
    args = setup_lm_eval_parser()
    model, _, tokenizer, generation_config = initialize_model(args, logger)

    if args.trust_remote_code:
        # trust_remote_code fix was introduced in lm_eval 0.4.3
        # https://github.com/EleutherAI/lm-evaluation-harness/pull/1998/files
        # We need to cherry-pick the fix manually untill we upgrade (SW-190418)
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    with torch.no_grad():
        lm = HabanaModelAdapter(tokenizer, model, args, generation_config)

    eval_start = time.perf_counter()
    with torch.no_grad():
        results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit_iters)
    if args.device == "hpu":
        import habana_frameworks.torch.hpu as torch_hpu

        torch_hpu.synchronize()
    eval_end = time.perf_counter()

    results["args"] = vars(args)
    results["duration"] = eval_end - eval_start

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
        json.dump(results, open(args.output_file, "w"), indent=2)
        if args.show_config:
            print(json.dumps(results, indent=2))
    if args.quant_config:
        finalize_quantization(model)
    if args.save_quantized_model_with_inc:
        save_model(model, tokenizer, args.saved_model_path)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
