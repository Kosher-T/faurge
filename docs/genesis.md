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

#### 🗄️ Raw Material Datasets
* **VCTK / LJSpeech:** Core baseline datasets for pristine, dry studio vocals.
* **VoicePersona-Dataset:** Ground-truth semantic metadata mapping physical acoustic textures to text (replaces manual tagging).
* **DAPS (Device and Produced Speech):** Parallel corpus containing pristine studio audio (`cleanraw`) perfectly time-aligned with the exact same speech recorded on consumer devices (smartphones, tablets) in real physical rooms.
* **MicIRP (Microphone Impulse Response Project):** ~65 physical Impulse Responses of vintage, rare, and classic microphones (1940s ribbons, Soviet dynamics, tube condensers).
* **MIT McDermott IRs:** High-quality physical environmental impulse responses (cathedrals, cars, classrooms).

---

#### 📌 Variables Legend
* **$V$**: Pristine, dry vocal segment (DAPS `cleanraw`, LJSpeech, VCTK).
* **$Desc$**: Semantic string describing the vocal's acoustic texture (sourced via VoicePersona-Dataset).
* **$IR_{tgt}$**: High-quality, labeled target Impulse Response (MIT dataset, or MicIRP for aesthetic upgrades).
* **$Scene$**: Semantic string describing the target room or target microphone (e.g., *"in a large brick open plan office"*, *"through a vintage 1950s ribbon microphone"*).
* **$IR_{bad}$**: Terrible, boxy, or phase-smeared Impulse Response.
* **$N$**: Noise arrays (HVAC, white noise, etc).
* **$IR_{lofi\_prop}$**: Destructive IR simulating physical props or vintage bandwidth limitations (sourced from MicIRP).

---

#### 🧮 The V3 Matrix

##### 1. Acoustic Space Transfer (The Room Swap) — 25% (5,000 samples)
* **Difficulty:** Extreme
* **Target (For Loss):** $V * IR_{tgt}$ (or DAPS `cleanraw` * $IR_{tgt}$)
* **Input 2 (Current STFT):** $(V * IR_{bad}) + N$ (or DAPS `iphone_bedroom` / consumer device recordings)
* **Input 1 ($c_{scene}$ Prompt):** `"[Desc] [Scene]."`
* **Logic:** Genesis must simultaneously strip a bad room, ignore the background noise, and synthesize the target room's decay tail.

##### 2. Acoustic Stripping (Dereverberation) — 20% (4,000 samples)
* **Difficulty:** Very High
* **Target (For Loss):** $V$ (Pure dry vocal)
* **Input 2 (Current STFT):** $(V * IR_{bad}) + N$ (Heavily leveraging DAPS real-world consumer recordings here)
* **Input 1 ($c_{scene}$ Prompt):** `"[Desc] in a completely dry, soundproof recording studio."`
* **Logic:** Reverses real-world comb-filtering, hardware clipping, and phase cancellation back to absolute zero.

##### 3. Harmonic Inpainting (Bandwidth Extension) — 20% (4,000 samples)
* **Difficulty:** Very High
* **Target (For Loss):** $V * IR_{tgt}$
* **Input 2 (Current STFT):** `Highpass(Lowpass( ` $V * IR_{tgt}$ ` ))` 
* **Input 1 ($c_{scene}$ Prompt):** `"A full-spectrum, high-fidelity [Desc] [Scene]."`
* **Logic:** Generates missing sub-harmonics (chest resonance) and high frequencies (air) from a severely frequency-truncated signal.

##### 4. Signal Extraction (Pure Denoising / Babble Removal) — 15% (3,000 samples)
* **Difficulty:** High
* **Target (For Loss):** Source audio before environmental playback.
* **Input 2 (Current STFT):** Distant-mic audio with heavy background bleed (Strictly sourced from VOiCES).
* **Input 1 ($c_{scene}$ Prompt):** `"[Desc] [Scene] with isolated, clear vocals."`
* **Logic:** The room stays exactly the same, but the fundamental frequency ($f_0$) of the primary speaker is preserved while convolutional background babble is phase-cancelled.

##### 5. Timbral & Proximity Reconstruction — 10% (2,000 samples)
* **Difficulty:** Medium
* **Target (For Loss):** $V * IR_{vintage\_mic}$ (Leveraging MicIRP to provide rich, warm harmonic targets)
* **Input 2 (Current STFT):** `Bad_Random_EQ( ` $V * IR_{tgt}$ ` )` or harsh headset microphone simulations.
* **Input 1 ($c_{scene}$ Prompt):** `"A rich, balanced, close-mic [Desc] [Scene]."`
* **Logic:** Corrects severe EQ imbalances (thin/tinny or booming proximity effect) and replaces harsh digital capture with warm analog harmonics.

##### 6. Stylistic / Diegetic Degradation — 5% (1,000 samples)
* **Difficulty:** Low
* **Target (For Loss):** $V * IR_{lofi\_prop}$ (MicIRP vintage broadcast mics, Walkie-Talkies, etc.)
* **Input 2 (Current STFT):** $V * IR_{pristine\_studio}$
* **Input 1 ($c_{scene}$ Prompt):** `"[Desc] transmitted through an antique 1940s dynamic microphone."`
* **Logic:** The reverse Faurge. Mathematically destroying a perfect vocal to match a cinematic diegetic prompt.

##### 7. Target Scene Projection (Acoustic Addition) — 5% (1,000 samples)
* **Difficulty:** Trivial
* **Target (For Loss):** $V * IR_{tgt}$
* **Input 2 (Current STFT):** $V$ (or $V$ with extremely mild background noise)
* **Input 1 ($c_{scene}$ Prompt):** `"[Desc] [Scene]."`
* **Logic:** Pure spatial synthesis from a sterile starting point, ensuring she explicitly maps the CLAP room/mic prompt to the target acoustic fingerprint.