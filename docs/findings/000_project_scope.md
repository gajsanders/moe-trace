# Finding 000 — Project Scope

## Purpose

MoE Trace is a research and engineering project for Mixture-of-Experts (MoE) models.

The project studies expert routing during local model inference.

The first target platform is Apple Silicon.

The first runtime is MLX and MLX-LM.

MoE Trace does not replace an inference runtime.

MoE Trace records routing data, analyzes the data, and tests cache policies.

## Problem

An MoE model contains many experts.

The router selects a small number of experts for each token.

A system can keep some experts in fast memory.

Other experts can be in slower memory or storage.

If the system can predict which experts it will need, it can possibly reduce memory access and data transfer.

This can reduce inference latency.

However, a predictive cache is useful only if expert routing has sufficient structure.

MoE Trace must first measure this structure.

## Initial research question

The initial research question is:

> Does MoE expert routing contain enough temporal and statistical structure to support a cache policy that performs better than simple cache policies?

The project must answer this question before it develops a complex prediction model.

## Initial project stages

The initial project has these stages:

1. Run a supported MoE model on Apple Silicon.
2. Record the expert selections.
3. Normalize and validate the routing data.
4. Measure routing behavior.
5. Simulate simple expert cache policies.
6. Compare simple policies with an oracle policy.
7. Test predictive policies only if sufficient improvement is possible.
8. Test the results on more prompts and workloads.

## Initial project output

The initial version of MoE Trace should provide these functions:

- Record MoE expert routing.
- Store normalized routing traces.
- Measure expert frequency.
- Measure adjacent-token expert overlap.
- Measure routing concentration.
- Compare routing behavior between layers.
- Simulate expert cache policies.
- Compare measured cache policies with an oracle policy.
- Produce reproducible benchmark results.

The initial project does not need to modify MLX or Metal.

The initial project does not need to make inference faster.

Its main purpose is to measure whether a useful optimization opportunity exists.

## Decision rules

### Routing structure

Continue if the routing data has meaningful temporal or statistical structure.

Examples of useful structure are:

- repeated expert use between adjacent tokens;
- large differences between layers;
- expert usage that is not uniform;
- predictable expert transitions.

If routing is approximately random and uniform, stop the predictive-cache work.

The routing-analysis tools can still be published.

### Cache simulation

Compare simple cache policies with an oracle policy.

The oracle knows future expert selections.

Continue if there is a meaningful performance gap between a simple policy and the oracle.

If a simple policy performs close to the oracle, stop the predictive-cache work.

There is not sufficient headroom for a more complex policy.

### Predictive policy

Start with simple prediction methods.

Use a learned model only if simple methods leave sufficient unused performance.

Stop increasing model complexity if the additional method gives only a small improvement.

### Runtime integration

Runtime integration is outside the initial project.

Start runtime integration only if simulation results show a strong and repeatable improvement.

Stop the runtime work if the simulated improvement does not produce a useful improvement on real hardware.

## Possible later work

If the initial project finds a useful cache policy, later work can include:

- integration with an MLX MoE runtime;
- expert prefetch;
- SSD-backed expert storage;
- hardware-aware cache allocation;
- KV-cache scheduling;
- RAG resource scheduling;
- agent memory scheduling.

These items are not requirements for the initial project.

## Status

Project stage: Routing characterization

Current decision: Continue
