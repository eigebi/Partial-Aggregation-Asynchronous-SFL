# Partial-Aggregation Asynchronous Split Federated Learning

This repository contains the reference implementation of our asynchronous split federated learning (SFL) framework with partial aggregation for heterogeneous edge systems.

## Overview

Split federated learning (SFL) is a distributed training framework that combines model splitting with federated aggregation, making it suitable for resource-limited edge devices. This project focuses on **asynchronous SFL** under heterogeneous communication and computation conditions.

The core idea is to decouple the aggregation paces on the client and server sides:

- **Client-side aggregation** is performed over ready devices only.
- **Server-side aggregation** exploits all available server-side progress, including updates from unfinished devices.

This design aims to improve the **wall-clock delay–convergence tradeoff** in heterogeneous edge environments.

## Repository Structure

```text
.
├── profiler/              # Profiling data / utilities for computation and communication cost
├── async_env.py           # Heterogeneous asynchronous system environment
├── batch_stream.py        # Batch / mini-batch stream utilities
├── config.py              # Experiment configuration
├── datasets.py            # Dataset loading and preprocessing
├── dirichlet_split.py     # Non-IID data partition with Dirichlet distribution
├── engine.py              # Core asynchronous SFL training engine; comparison schemes implementations are stored here
├── main.py                # Main entry point for running experiments
├── models.py              # Model definitions and split-model utilities
├── .gitignore
└── README.md