# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import os

# %% Functions


def find_blocks(lines: list[str]) -> list[list[str]]:
    blocks = [[]]
    for line in lines:
        if line == "\n":
            blocks.append([])
        else:
            blocks[-1].append(line)
    return blocks


def replace_toctree(blocks: list[list[str]]) -> list[list[str]]:
    toctree_block_id = -1
    for i_block, block in enumerate(blocks):
        print("----------------------------")
        is_toctree = False
        for line in block:
            print(line, end="")
            if "toctree::" in line:
                toctree_block_id = i_block

    blocks[0] = []
    blocks[0].append("Summary\n")
    blocks[0].append("=======\n")
    # old version: just replace <toctree> by <autosummary>
    blocks[toctree_block_id] = []
    blocks[toctree_block_id].append(".. autosummary::\n")
    blocks[toctree_block_id].append("   :toctree: _autosummary_generated\n")
    blocks[toctree_block_id].append("   :recursive:\n")
    # blocks[toctree_block_id].append("   :template: ../_templates/apidoc/package.rst\n")

    return blocks


def clean_toctree(blocks: list[list[str]]) -> list[list[str]]:
    toctree_block_id = -1
    for i_block, block in enumerate(blocks):
        print("----------------------------")
        is_toctree = False
        for line in block:
            print(line, end="")
            if "toctree::" in line:
                toctree_block_id = i_block

    blocks[0] = []
    blocks[0].append("API\n")
    blocks[0].append("===\n")
    # new version: hide toctree + add autosummary
    blocks[toctree_block_id].append("   :hidden:\n")
    for i_line, line in enumerate(blocks[toctree_block_id]):
        if ":maxdepth:" in line:
            blocks[toctree_block_id][i_line] = "   :maxdepth: 1\n"

    return blocks


def read_lines(in_fname: str) -> list[str]:
    with open(in_fname, "r") as in_mfile:
        lines = in_mfile.readlines()
    return lines


def write_blocks(out_fname: str, blocks: list[list[str]]) -> None:
    with open(out_fname, "w") as out_mfile:
        for i_block, block in enumerate(blocks):
            for line in block:
                out_mfile.write(line)
            if i_block < len(blocks) - 1:
                out_mfile.write("\n")


def fix_module(fname: str) -> None:
    assert os.path.isfile(in_fname)
    parent_dir = os.path.dirname(fname)

    lines = read_lines(fname)

    for i_line, line in enumerate(lines):
        print("<<<{}>>>".format(line))
        if line == "\n":
            print("return line:", i_line)

    original_blocks = find_blocks(lines)
    blocks = clean_toctree(original_blocks)
    out_fname = fname
    write_blocks(out_fname, blocks)

    blocks = replace_toctree(original_blocks)
    out_fname = os.path.join(parent_dir, "modules_autosummary.rst")
    write_blocks(out_fname, blocks)


# %% Main Script
if __name__ == "__main__":
    for package_name in ["plaid", "tests", "examples"]:
        in_fname = os.path.join("api_reference", package_name, "modules.rst")
        fix_module(in_fname)
