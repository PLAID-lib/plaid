# Dataset command-line tools

`plaid-tools` extracts, merges, and repartitions local PLAID datasets. It can
also be run as `python -m plaid.cli.tools`. Use `--help` on any subcommand.

Every operation creates a **new dataset**; an existing output path is rejected,
even if empty. Sources are never modified. Output is staged in a temporary
sibling directory and published after writing succeeds. Failed writes are
cleaned up.

## Output backend

All commands accept `--backend cgns`, `--backend hf_datasets`, or `--backend zarr`.
This overrides only the output format: source backends are detected automatically.
Without an override, the source backend is preserved (the first source for merges).
Sample counts, schemas, constants, and backend metadata are regenerated.

## Subset

```bash
plaid-tools subset SOURCE OUTPUT --select train --select test:0,2,5:10
plaid-tools subset SOURCE OUTPUT --select train:0:100 --backend zarr
```

Repeat `--select` for each split or sample selection. A split name selects all its
samples. After the first colon, use comma-separated zero-based indices or
stop-exclusive slices (`start:stop[:step]`), with omitted bounds allowed. Only
nonnegative, in-bounds indices and positive steps are accepted. Selected order is
preserved within each split; output indices start at zero. Repeated samples and
empty selections are rejected. Omitting `--select` copies every split and sample.

## Merge

```bash
plaid-tools merge SOURCE_A SOURCE_B SOURCE_C OUTPUT --backend hf_datasets
```

At least two inputs are required. Inputs can have different storage backends.
Same-named splits are concatenated in input order; other splits are retained.
Samples are not deduplicated. Compatible sample structures are required by the
chosen output backend; conversion or schema failures leave no published output.

By default, descriptive infos (owner, license, data production, and description)
must match. Sample counts and backend names need not match. Use `--metadata first`
to explicitly accept differences and retain the first input's descriptive infos.
This does not reconcile licenses or establish permission to combine datasets.

## Partition

```bash
plaid-tools partition SOURCE OUTPUT --ratios train=0.8 test=0.2 --seed 42
plaid-tools partition SOURCE OUTPUT --select train \
    --counts train=80 validation=20 --no-shuffle --backend cgns
```

All source samples are pooled by default. `--select` has the same syntax as for
subset; unselected samples are not copied. Choose either positive ratios summing
to one or positive integer counts summing to the number of selected samples.
Ratios use largest-remainder rounding (ties follow argument order), so every
selected sample appears exactly once. Every output split must be nonempty.
Output split names must be unique and cannot contain `/`, `\`, or `:` or equal
`.` or `..`.

Samples are shuffled using a local NumPy generator with the nonnegative seed
`42` by default. Supply `--seed` for reproducibility or `--no-shuffle` to preserve
selection order. Named output splits replace the original split layout.

## Problem definitions

Problem definitions are **not copied** by these commands. A warning is emitted
for each source containing definitions, because split names and sample indices
may change. Recreate definitions for the new dataset rather than retaining stale
references. Other arbitrary source files (such as dataset cards) are not copied.