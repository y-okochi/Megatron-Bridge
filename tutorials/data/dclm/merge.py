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

import os
import shlex
import argparse
import subprocess


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_save", type=str, required=True, help="Path where to save merged file.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to decompressed dataset.")

    return parser


def merge_data(
    path_to_save: str,
    source_dir: str,
) -> None:
    cmd = (
        f'cd {source_dir} && '
        f'cat *.jsonl > {path_to_save} && '
        'rm shard_*'
    )

    print("Merging files...")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == '__main__':
    args = arguments().parse_args()

    merge_data(
        path_to_save=args.path_to_save,
        source_dir=args.source_dir,
    )
