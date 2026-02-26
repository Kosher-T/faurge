# Agent 3: Genesis (The Generative Math Engine & DDSP Actuator)

## Part 1: Overview
Genesis is the acoustic architect and harmonic reconstructor of Faurge. While standard digital signal processing (Agent 2: Ursula) can tweak the volume of specific frequencies, it cannot repair inharmonic distortion (like a crackling cheap microphone) or create physical space. Genesis handles the abstract, generative math required to synthesize reality.

Operating strictly as a **One-Shot Actuator** within the Orchestrator's closed-loop system, Genesis does not evaluate her own work or perform live Reinforcement Learning. She receives a mathematical target coordinate representing the desired room and vocal texture, reads the raw audio's spectrogram, and executes a single forward pass. 

She mathematically compiles this data into a custom Impulse Response (`ir`)—a literal audio file that bridges the harmonic gap between a degraded microphone and a professional studio—and immediately unloads from VRAM to preserve the 4GB system budget.

---

## Part 2: Engineering Specification

### 1. Core Architecture (DDSP Synthesizer)
Genesis is built on a **DDSP (Differentiable Digital Signal Processing)** architecture. Standard generative audio models (like Latent Diffusion) attempt to draw raw audio waveforms pixel-by-pixel, which requires massive VRAM and induces heavy latency. 

Instead, Genesis's neural network outputs *parameters* for mathematical oscillators and noise synthesizers. This allows her to generate high-fidelity acoustic environments and rebuild missing vocal harmonics at a fraction of the computational cost.

### 2. The Inputs (The Abstract State)
Genesis does not read physical numbers like LUFS or LTAS. She operates entirely in latent geometry and frequency-time matrices.

* **Input A (The Desired Scene):** The accumulated latent target passed down from the Orchestrator. 
  * $c_{scene} \in \mathbb{R}^{512}$ (The literal CLAP vector representing the acoustic environment and vocal smoothness).
* **Input B (The Current Reality):** The High-Resolution STFT (Short-Time Fourier Transform) of the *raw* audio buffer currently in the Shadow Space. 
  * This allows her network to explicitly "see" the exact time-slices where inharmonic distortion (crackle) or phase cancellation occurs.

### 3. Processing Mechanics (One-Shot Execution)
1. Genesis receives the inputs and utilizes **Cross-Attention** (or FiLM layers) to condition the raw STFT matrix with the $c_{scene}$ vector. This teaches the network *where* the audio needs to go.
2. She performs a **single forward pass**. Her neural network calculates the exact harmonic amplitudes, noise envelopes, and reverb tail characteristics required to delete the bad acoustics and build the requested scene.
3. The DDSP synthesizer compiles these calculated parameters into a pristine audio file.
4. She immediately unloads from VRAM.

### 4. Raw Outputs (`ir`)
Genesis outputs a literal `.wav` file containing the Impulse Response (`ir`). 

*Note: In Round 2 of the reflection loop, if Fabian dictates that the audio is still flawed, Genesis does not "tweak" her previous IR. She receives the newly accumulated $c_{scene}$ from the Orchestrator and mathematically synthesizes a brand new `ir.wav` from scratch.*

---

### 5. Cloud Training Methodology (The Inverse-Acoustic Sandbox)

To function as the ultimate harmonic reconstructor, Genesis must be trained in the cloud to understand the exact mathematical inverse of a degraded microphone. This is achieved using a heavily curated, synthetic dataset designed to fit within Kaggle's 20GB output limits.

#### A. The Core Forward Pass & Convolution Loss
Genesis is not trained to directly guess an `ir` waveform, as phase alignment makes that mathematically impossible to learn. Instead, the loop evaluates the *effect* of her IR:
1. **The Target:** A clean, dry vocal is convolved with a target IR to create the perfect Ground Truth audio. The text description of this target room is encoded into the $c_{scene}$ vector.
2. **The Input:** The same dry vocal is subjected to aggressive degradation (e.g., bitcrushing, white/pink noise, clipping, highpass/lowpass filtering). The STFT of this ruined audio is extracted.
3. **The Prediction:** Genesis receives $c_{scene}$ and the ruined STFT, and attempts to predict the DDSP parameters to generate an `ir`.
4. **The Loss Function:** Her generated `ir` is convolved with the *ruined* input audio to create a proposed output ($\\hat{S}$). This output is compared against the true pristine audio ($S$) using **Multi-Scale Spectral Loss ($L_{MSS}$)**:
   $$L_{MSS} = \sum_{i} \left( ||S_i - \hat{S}_i||_1 + \alpha ||\log S_i - \log \hat{S}_i||_1 \right)$$

By backpropagating this error, she learns exactly how to manipulate the DDSP oscillators to neutralize a bad room and perfectly reconstruct the harmonic series.

#### B. The 20,000 Sample Dataset Matrix
To ensure Genesis can handle all real-world failures, the dataset generates exactly 20,000 training triples, rigidly distributed across five core acoustic repair scenarios:

**1. Signal Extraction & Room Dynamics (20% | 4,000 samples)**
*Teaches her how noise interacts with acoustic space.*
* **1c (6% | 1,200 samples):** *Target:* Room A | *Input:* Room B + Noise. (The hardest calculation: strips a bad room, strips noise, synthesizes a new room).
* **1 (7% | 1,400 samples):** *Target:* Room A | *Input:* Room A + Noise. (Baseline control: deletes noise without deleting the existing reverb tail).
* **1a (4% | 800 samples):** *Target:* Dry | *Input:* Room A + Noise. (Total acoustic stripping).
* **1b (3% | 600 samples):** *Target:* Room A | *Input:* Dry + Noise. (Adds room while filtering out noise).

**2. Harmonic Inpainting (20% | 4,000 samples)**
* **2 (20% | 4,000 samples):** *Target:* Full Spectrum + Room A | *Input:* Telephone Bandwidth + Room A. (Forces the network to generate missing chest and air frequencies from heavily truncated signals).

**3. Acoustic Stripping / Dereverberation (20% | 4,000 samples)**
* **3a (13% | 2,600 samples):** *Target:* Room A | *Input:* Muddy Room. (The ultimate "Room Swap", reversing heavy phase cancellation).
* **3 (7% | 1,400 samples):** *Target:* Dry | *Input:* Muddy Room. (Reverses comb-filtering back to a pristine zero-state).

**4. Target Projection (20% | 4,000 samples)**
* **4 (20% | 4,000 samples):** *Target:* Specific IR | *Input:* Dry. (Teaches the DDSP engine to map the CLAP geometry directly to exact early reflections and decay times).

**5. Pure Denoising / Crackle Removal (20% | 4,000 samples)**
* **5 (20% | 4,000 samples):** *Target:* Dry | *Input:* Dry + Noise/Crackle. (Foundational cleanup. Differentiates the human voice fundamental frequencies from electrical hums, broadband hiss, and hardware pops).