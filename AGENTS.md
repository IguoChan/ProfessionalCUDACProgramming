# Repository Guidelines

## Project Structure & Module Organization

This repository contains CUDA C examples organized by book chapter under `code_samples/`.

- `code_samples/chapter_01/`, `chapter_02/`, and `chapter_03/` contain standalone sample programs and a chapter-local `Makefile`.
- `code_samples/common/common.h` contains shared CUDA error-checking macros and timing helpers.
- Sample executables are built next to their source files. Run `make clean` in the chapter directory before committing build outputs.

Add new examples to the most relevant `code_samples/chapter_##/` directory. Keep each sample self-contained unless a helper is broadly reusable, in which case place it in `code_samples/common/`.

## Build, Test, and Development Commands

Run commands from the chapter directory you are working in:

```sh
cd code_samples/chapter_02
make
```

Builds all programs listed in that chapter's `Makefile` using `nvcc` for `.cu` files and `gcc` for `.c` files.

```sh
make clean
```

Removes generated executables for that chapter.

```sh
./sumArraysOnGPU-timer
```

Runs a compiled sample. Use the exact executable name built by the local `Makefile`.

The current Makefiles target `-arch=sm_89`; adjust only when the repository needs to support a different GPU architecture.

## Coding Style & Naming Conventions

Follow the existing C/CUDA style: 4-space indentation, braces on their own line for functions and control blocks, and simple descriptive names such as `sumMatrixOnGPU2D`, `checkResult`, and `initialData`. Use `.cu` for CUDA examples and `.c` for host-only C examples. Include shared helpers with `#include "../common/common.h"` from chapter directories.

Prefer explicit CUDA error checks using `CHECK(...)` around runtime API calls. Keep comments focused on what the sample demonstrates.

## Testing Guidelines

There is no separate test framework. Validation is done by building the relevant chapter and running the affected executable. For numerical examples, preserve or add host-vs-GPU result checks and clear failure output. Before submitting, run:

```sh
make clean && make
./affected_sample
```

## Commit & Pull Request Guidelines

The Git history currently contains only an initial commit, so use concise imperative commit messages, for example `Add chapter 03 divergence sample` or `Fix matrix bounds check`.

Pull requests should describe the changed sample, the GPU/CUDA environment used, and the commands run for verification. Link related issues when available and avoid including generated executables or intermediate CUDA artifacts.
