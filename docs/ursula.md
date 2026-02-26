# Agent 2: Ursula (The DSP Specialist & Actuator)

## Part 1: Overview
Ursula is the "Fast Path" actuator of the Fa-Ur-Ge pipeline. While Fabian thinks in semantics and Genesis thinks in latent geometry, Ursula operates strictly in the physical reality of standard digital signal processing (DSP). 

She is completely blind to English text, CLAP vectors, and the overarching reflection loop. She does not know what a "cheap microphone" is; she only knows that the current audio has too much 500Hz and needs to be compressed. 

Critically, Ursula operates as a **One-Shot Executioner** during live local inference. She does not perform Reinforcement Learning (RL) or iterative guessing on the edge device. She takes the physical targets provided by the Orchestrator, performs a single forward pass through her neural network, outputs the exact C++ plugin parameters, and immediately unloads from VRAM.

---

## Part 2: Engineering Specification

### 1. Core Architecture
Ursula's brain is a highly optimized **Policy Network** (a Multi-Layer Perceptron). 
* **Cloud Phase:** During training, this network acts as the Actor in a Soft Actor-Critic (SAC) Reinforcement Learning setup.
* **Edge Phase (Live):** During local execution, the RL components are stripped away. She operates as a frozen, deterministic feed-forward network.

### 2. The Inputs (The Dual State)
Ursula receives no semantic data and no Purity/Distortion metrics (as standard EQ/Compression cannot fix inharmonic distortion). She receives two distinct arrays from the Orchestrator:

* **Input A (The Current Reality):** The physical metrics of the audio buffer currently in the Shadow Space.
  * $C_{LTAS} \in \mathbb{R}^{64}$
  * $C_{LUFS} \in \mathbb{R}$
  * $C_{Dyn} \in \mathbb{R}^2$
* **Input B (The Accumulated Target):** The absolute physical goals calculated by Fabian and accumulated by the Orchestrator ($T_{new} = T_{n-1} + \Delta T_n$). Ursula never sees the Deltas; she only sees the final absolute number she needs to hit.
  * $T_{LTAS} \in \mathbb{R}^{64}$
  * $T_{LUFS} \in \mathbb{R}$
  * $T_{Dyn} \in \mathbb{R}^2$

### 3. Processing Mechanics (One-Shot Execution)
1. Ursula concatenates the Current Reality and the Accumulated Target into a single input tensor.
2. The tensor passes through her hidden layers. Because she was trained via RL to understand how DSP knobs affect audio, her network calculates the exact parameter shifts required to bridge the gap between $C$ and $T$.
3. The output layer applies `tanh` activations to scale the raw outputs into normalized boundaries (e.g., mapping a neural output of `[-1, 1]` to an EQ gain range of `-24dB` to `+24dB`).

### 4. Raw Outputs (`dsp_state`)
Ursula does not output audio. She outputs a highly structured dictionary of floating-point numbers representing the API parameters for standard headless Linux DSP plugins (LV2/LADSPA).

```json
{
  "eq_band_1_freq": 120.0,
  "eq_band_1_gain": -4.5,
  "eq_band_2_freq": 500.0,
  "eq_band_2_gain": -2.1,
  "comp_threshold": -18.5,
  "comp_ratio": 4.0,
  "deesser_threshold": -22.0
}