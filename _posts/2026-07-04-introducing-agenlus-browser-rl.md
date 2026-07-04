---
layout: post
title: "Show HN: Agenlus – Training and Competing with RL Agents in the Browser"
date: 2026-07-04 20:00:00 +0900
categories: [Projects]
tags: [reinforcement-learning, webgpu, webassembly, machine-learning, show-hn]
---

## A Quick Primer: What is Reinforcement Learning (RL)?

If you have mostly worked with Large Language Models (LLMs) or Supervised Learning, RL is a shift in mindset:

* **Supervised Learning** (like predicting the next token in a text file or classifying an image) relies on a static dataset of static correct answers.
* **Reinforcement Learning** relies on **active feedback loops**. An **Agent** (our AI model) interacts with an **Environment** (the game or simulation). It takes an **Action**, receives an **Observation** (the new state of the environment) and a **Reward** (a signal telling the agent how well it is doing), and repeats.

The agent’s goal is to learn a **Policy** (a mapping from observations to actions) that maximizes the cumulative reward over time. It starts completely random and improves purely through trial and error.

Because there is no "correct answer" provided upfront, RL agents often find incredibly clever, emergent ways to solve games that developers never anticipated. However, this trial-and-error process is computationally intensive and requires millions of interactions, which is why bringing it directly to the browser is both challenging and exciting.

---

Reinforcement Learning (RL) has always been one of the most fascinating branches of AI. There is something deeply satisfying about watching a blank-slate agent explore an environment and gradually emerge with a superhuman policy. 

Yet, compared to the explosive growth of LLM playgrounds and tools, RL remains relatively inaccessible. Setting up a local environment often means wrestling with Python virtual environments, CUDA versions, PyTorch installations, and headless rendering bugs in Gymnasium. 

We built **[Agenlus](https://agenlus.com)** to solve this. It is a community platform and model hub for Reinforcement Learning designed to run **entirely in the browser**—no installation, no CUDA configuration, just instant training and evaluation.

### See It in Action

Here is a quick YouTube demonstration of the platform in action:

<a href="https://www.youtube.com/watch?v=-ewt-BV23D0" target="_blank" style="display: block; width: 100%;">
  <img src="https://img.youtube.com/vi/-ewt-BV23D0/0.jpg" alt="Agenlus Demo Video" style="width: 100%;" />
</a>

And here is a preview of the web UI running a client-side training loop:

<img src="/image.png" alt="Agenlus Web UI" style="width: 100%;" />

Here is how we built the architecture to make in-browser RL viable, the UX challenges we encountered, and why we believe RL is prime for a B2C comeback.

---

## The Architecture: WebGPU + Pyodide + Web Worker

To achieve a zero-install experience, we had to move both the environment simulation and the model training into the client browser. 

Our core architecture splits the load into three parts:

```
Browser Context
├── Web Worker (Pyodide)
│   ├── gymnasium (micropip)
│   ├── env.step(action) -> observation, reward, terminated
│   └── Sends state updates to the main thread
│
├── Main Thread (WebGPU & JS)
│   ├── Model inference & policy updates
│   ├── Action selection (via WebGPU-accelerated tensors)
│   └── Direct HTML5 Canvas drawing (replacing Pygame rendering)
```

### 1. Pyodide in a Web Worker
We use [Pyodide](https://pyodide.org/) (Python compiled to WebAssembly) to run official `gymnasium` environments like `CartPole` and `MountainCar`. 
Running Python environments in the browser can easily freeze the UI because Python execution is single-threaded. By offloading Pyodide to a dedicated **Web Worker**, we keep the main UI thread running at a smooth 60fps.

### 2. Main Thread for WebGPU Inference & Canvas Rendering
We utilize **WebGPU** on the main thread for neural network updates and forward passes. 
Because the observation state returned by Pyodide is simple (e.g., 4 float values for `CartPole`), passing data between the Web Worker and the WebGPU context has negligible serialization overhead.

Furthermore, we completely bypassed Pygame's rendering system. Pygame cannot easily output to a browser canvas without heavy overhead. Instead, we read the raw state values from Pyodide and draw the environment using native **HTML5 Canvas**. This not only looks much cleaner and more modern, but it also runs orders of magnitude faster than sending raw frame buffers from WASM.

---

## Overcoming Browser Constraints

Building a fully client-side ML training loop comes with unique friction points. Here is how we addressed them:

### 1. The 15-Second Pyodide Cold Start
Loading Pyodide, initializing the Python runtime, and installing dependencies via `micropip` (like `numpy` and `scipy`) can take anywhere from 10 to 30 seconds depending on the user's connection. 
To hide this latency:
* **Background Preloading**: We spin up the Web Worker and start loading Pyodide in the background the moment a user lands on the platform, even while they are just browsing the catalog of environments.
* **Service Worker Caching**: Once loaded, WASM binaries and pip packages are aggressively cached via IndexedDB/Cache API so subsequent visits load instantly.

### 2. Session Persistence (Handling Tab Closes)
Unlike server-side training, if a user closes their browser tab, they risk losing their active training progress. 
Currently, we save basic training metadata (such as total episodes, training steps, and best reward) as well as local demonstration logs to the browser's **localStorage** for progress tracking. To solve the problem of losing active model states, our immediate next step is to implement automatic agent checkpointing (saving model weights and hyperparameters) to **IndexedDB**, allowing users to seamlessly resume active training sessions even after a tab close or crash.

---

## The Core Loop: Imitation Learning as a Gateway

A major roadblock in RL is sample inefficiency: an agent can take thousands of episodes just to learn how to balance a pole. For a casual user in a browser, waiting 10 minutes for *any* sign of intelligence is a retention killer.

We designed a dual-mode interaction loop to fix this:

1. **Human Demonstration (Direct Control)**: Users can directly play the game using their keyboard/mouse to generate demonstration trajectories.
2. **Imitation Learning (Behavioral Cloning)**: We use these human demonstrations to train a base model instantly. The agent immediately mirrors the player's playstyle.
3. **RL Fine-Tuning**: Once the agent has a basic grasp of the task, the user hands over control to the RL algorithm (PPO, DQN) to optimize the policy beyond human performance.

This "human-in-the-loop" approach solves the cold-start problem of training from scratch and gives users a strong sense of ownership: *"This agent is learning to play like me, but better."*

---

## Why B2C RL is Highly Viable Today

We are seeing immense compute inflation with LLMs, making B2C AI startups highly dependent on massive API bills or VC funding. 

RL models, especially for classic control, platformers, and simple 2D/3D environments, are computationally lightweight. A small MLP or a tiny CNN is often enough to achieve state-of-the-art performance in these games. 
Because training runs entirely on the **user's client hardware via WebGPU**, our infrastructure costs are virtually zero. This allows us to offer a free tier without burning through cloud credits, shifting the monetization focus to custom-made environments and premium model exports.

---

## What’s Next?

Our current roadmap focuses on:
* **Multi-Agent PvP & Co-op**: Allowing users to upload their trained agents to fight or cooperate with other users' agents in real-time.
* **Low-Code Environment Builder**: A Roblox-style studio allowing users to customize and upload their own Gymnasium-compatible environments.

We would love to get your feedback on our architecture and approach. How would you design client-side RL differently? What environments would you like to see trained in a browser?

Check out the project: **[Agenlus.com](https://agenlus.com)**
