---
layout: page
title: About me
permalink: /about/
---

# Hi there, I'm YoungSeong Kim

**Independent AI Researcher & Builder**

## About Me

I'm an independent researcher who believes that the most valuable contributions in AI come from building principled theoretical frameworks — ones that don't just explain existing phenomena, but actively guide the design of new methods. My work sits at the intersection of theory and methodology: I care as much about *why* something works as *how* to make it work better.

The same philosophy carries into how I build. I'm not interested in shipping yet another wrapper or fine-tuned model — I look for the gaps that others treat as given, and build from there. That's what led to Agenlus: while most ML tooling orbits around LLMs and supervised learning, RL remained oddly underserved as a community. Interesting RL environments like MineRL have historically demanded enormous compute and setup overhead just to get started. Agenlus flips that: simple rules, expressive environments, and everything runs in the browser — no cluster required.

---

## Research

### [Window is Everything (2025)](https://zenodo.org/records/17103133)

> 17K+ views · 4K+ downloads · Hacker News Front Page (Sep 2025)

A unified theory of neural operations that decomposes any operation into three orthogonal components:

- **Path (P)**: Defines operational locality — what the operation "sees"
- **Shape (S)**: Encodes geometric structure and symmetry — how it sees
- **Weight (W)**: Determines feature importance — what it emphasizes

**Key Contributions:**

- **Principle of Structural Alignment**: Optimal generalization arises when the structure of an operation matches the structure of the data
- **Information Bottleneck Connection**: Formal grounding of the framework in information theory
- **Complexity Dichotomy**: Distinguishes static capacity from adaptive regularization across operation types
- **Generative Framework**: A systematic pathway from observed data properties to principled architecture design

---

## Projects

### [Agenlus](https://agenlus.com)

A community platform for reinforcement learning — think HuggingFace's model hub, but built around RL environments and agents.

**Core features:**

- Browser-based RL training via **WebGPU + Pyodide** — no install, no setup, just train
- **Env Builder IDE**: write, test, and publish custom Gymnasium environments directly in the browser
- **Leaderboards** and model sharing across the community
- Fully serverless architecture on **Cloudflare Workers**

The goal: lower the barrier to RL experimentation the same way HuggingFace lowered the barrier to sharing models.

### [FactorizedAttention](https://github.com/Kim-Ai-gpu/FactorizedAttention)

PyTorch implementation of Factorized Attention mechanisms, derived from the P/S/W decomposition framework.

---

## Tech Stack

Python · PyTorch · NumPy · JAX · JavaScript · Cloudflare

---

## Get in Touch

- Email: [dafaafafaf33@gmail.com](mailto:dafaafafaf33@gmail.com)
- Twitter / X: [@salam341353](https://x.com/salam341353)
- GitHub: [Kim-Ai-gpu](https://github.com/Kim-Ai-gpu)
