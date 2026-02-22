# Faurge: The Generative Acoustic Reality Engine

> **Development is currently for Linux OS only. Windows and MacOS versions coming soon.**

## Overview
Faurge is a high-performance, agentic AI suite designed to transform raw audio signals into idealized acoustic states. Built on a "Split-Brain" architecture, Faurge bridges the gap between complex neural reasoning and low-latency digital signal processing (DSP). It functions as a software-defined vocal tract and acoustic environment, capable of correcting hardware deficiencies, neutralizing room acoustics, and synthesizing professional-grade audio textures.

---

## Deployment and Execution: The "Bake and Sleep" Model
Faurge operates natively on Linux (via PipeWire) as a highly efficient, resource-aware system utility. It is designed with a strict "Zero-Idle Overhead" philosophy to accommodate high-demand applications like gaming and 3D rendering. **The AI agents never process live audio.**

1. **Initialization**: Faurge hooks into the system audio server, establishing a virtual patch bay between physical hardware and software applications using standard, headless C++ DSP plugins.
2. **The Shadow Space (Asynchronous Capture)**: When triggered, Faurge captures a brief, 5-second buffer of the live audio into a non-audible parallel environment and wakes the AI agents.
3. **The Bake**: The agents analyze the buffer, calculate the optimal DSP parameters, and compile static Impulse Responses (IRs). They "bake" these solutions into the active PipeWire plugins, and immediately terminate to free up VRAM. 
4. **Execution**: The dumb, lightweight C++ plugins apply the baked acoustic transformation to the live audio stream in real-time with `<1%` CPU footprint.

---

## Architecture: The Tensor Network
Faurge Abandons semantic communication between agents in favor of high-dimensional latent space embeddings and concrete acoustic physics. The system is orchestrated by three specialized agents (Fa-Ur-Ge) working in a unified mathematical pipeline:

* **Fabian (Agent 1 - The Semantic Router)**: A lightweight NLP model built on a frozen CLAP text encoder. Fabian takes the user's semantic prompt and generates a **Split Destination**: concrete physical metrics (LTAS, LUFS, Crest Factor) for the source, and an abstract mathematical vector (CLAP) for the scene.
* **Ursula (Agent 2 - The DSP Specialist)**: Operating within a Reinforcement Learning loop, Ursula completely ignores semantic text. She strictly compares Fabian's physical targets (LTAS/LUFS) against the audio's current metrics. She manipulates standard headless DSP nodes (EQ, Compression, De-essing) to minimize the physical distance between the mic's raw state and Fabian's target physics.
* **Genesis (Agent 3 - The Math Engine)**: A Differentiable DSP (DDSP) model utilizing High-Res STFT math and a Harmonic plus Noise synthesizer. Genesis bridges the gap between Fabian's abstract scene vector and the EQ'd audio from Ursula. She synthesizes exact Impulse Responses and Procedural Comfort Noise without drawing raw audio waveforms, naturally smoothing jagged frequencies in the process.

---

## Training Data and Methodology
Faurge’s intelligence is derived from a multi-modal, synthetic training strategy:

* **Ursula's RL Sandbox**: Ursula learns the physics of audio manipulation by interacting with a custom Gymnasium environment, tuning headless DSP plugins on dry vocals (VCTK/LibriSpeech) to hit randomized LTAS/LUFS targets.
* **Genesis's DDSP Pairing**: Genesis learns to translate semantic CLAP vectors (from LAION-Audio-630K) into physical acoustic space by minimizing the Multi-Scale Spectral Loss against real-world Impulse Responses.
* **Physical Regression**: Fabian's routing and translation heads are trained to perfectly map English adjectives to hard audio physics and binary execution flags.

---

## Primary Use Cases

### 1. Asynchronous Gaming Audio Optimization
Faurge isolates and records specific application audio (e.g., footstep frequencies) during heavy loads. Once VRAM is freed, Faurge asynchronously analyzes the sample, generates a custom DSP profile, and seamlessly applies it to the active game state without sacrificing frame rates.

### 2. Acoustic Seeding and Comfort Noise Generation
For hardware that records heavily gated or "dead" audio, Genesis mathematically synthesizes an "Acoustic Bed" (e.g., warm studio air, subtle room tone) beneath the vocal track. This eliminates listener fatigue and mimics the environmental texture of high-end condenser microphones.

### 3. Professional Broadcast from Uncontrolled Environments
Faurge mathematically "deletes" the sound of a resonant bedroom, replacing the acoustics with a synthesized "Studio" environment in real-time.

### 4. Hardware Compensation and Vocal Reshaping
The system analyzes the frequency response of budget microphones and generates a custom inverse-filter. Utilizing DDSP harmonic reconstruction, Faurge naturally drops inharmonic distortion and rebuilds the vocal tract with silky precision.