# tf-dynamics-op

This repository contains source code for TensorFlow ops to calculate dynamics-based loss functions to train neural networks to produce efficient and feasible robot trajectories.

## Requirements

- Linux + a C++ compiler (supporting at least C++14)
- TensorFlow (version 2.7.0 is known to work)
  - must have been compiled using `-D_GLIBCXX_USE_CXX11_ABI=1` for compatiblity with pinocchio
- LAPACK
- [pinocchio](https://github.com/stack-of-tasks/pinocchio) (version 2.5.6 is known to work)

## Building

Run `make` in the main directory of this repository. Despite the low number of files, this can take a while.

## Usage

`dynamics.py` shows exemplary usage of the ops wrapped in a function with custom gradient.
