# Agent 3: Genesis (The Generative Math Engine)

## Part 1: Overview
Genesis is the acoustic architect of Faurge. While standard digital signal processing (DSP) tools can tweak the frequencies of a voice, they cannot create the physical space around it. When a microphone sounds unnatural because it is entirely devoid of background space (like an aggressively noise-canceled headset), or when I need to place a recording into a specific physical environment (like a tiled room or a broadcast studio), Genesis steps in.

She does not manipulate sliders. She generates reality. 

As the final stage in the Fa-Ur-Ge pipeline, Genesis receives the pristine, EQ'd audio from Agent 2 (Ursula) and the semantic acoustic destination from Agent 1 (Fabian). Using complex signal generation algorithms, Genesis synthesizes mathematical Impulse Responses (IRs) to build high-fidelity acoustic rooms. 

Furthermore, because absolute digital silence causes psychoacoustic discomfort (listener fatigue), she utilizes procedural algorithms to generate "Comfort Noise"—subtle, infinite background textures like studio air, HVAC hum, or distant traffic—to anchor the voice in a believable physical reality.

---

## Part 2: Engineering Specification

### 1. Core Architecture (DDSP Synthesizer)
Genesis is built on a **DDSP (Differentiable Digital Signal Processing)** architecture. Standard generative audio models (like Latent Diffusion) attempt to draw raw audio waveforms pixel-by-pixel. That requires massive VRAM and introduces terrible phase-blurring. 

I designed Genesis to bypass the Curse of Dimensionality. Her neural network outputs control parameters for a hard-coded **Harmonic plus Noise Synthesizer**. 


The math defining her acoustic generation is:
$$y(t) = \sum_{k=1}^K A_k(t) \sin(\phi_k(t)) + (h * n)(t)$$
*(Where the first term generates the resonant ringing/harmonics of a room, and the second term is filtered white noise representing the chaotic scattering of sound waves).*

This guarantees pristine, artifact-free audio generation with an ultra-low footprint that easily respects my 4GB VRAM limit.

### 2. The Inputs
Genesis requires a dual-input conditioning architecture to bridge the gap between Fabian's abstract semantic meaning and the physical mechanics of sound.

* **Input A: The Steering Wheel (CLAP Vector)**
  * **Source:** Fabian (Agent 1).
  * **Structure:** $c_{scene} \in \mathbb{R}^{512}$
  * **Function:** A 1D latent space embedding containing the global "meaning" of the target environment. It tells Genesis what the space *is*, but contains no phase or timing data.

* **Input B: The Raw Material (STFT Matrix)**
  * **Source:** The Shadow Space (Post-Ursula processing).
  * **Structure:** $X(m,k) \in \mathbb{C}^{1025 \times N}$
  * **Function:** The Short-Time Fourier Transform of the current audio. By resolving the signal into math, Genesis accesses the exact magnitude and phase of the frequencies. She needs this structural blueprint to prevent destructive interference (phase-cancellation) when generating new acoustics around the vocal.

### 3. Processing Mechanics & FiLM Integration
I cannot directly add or concatenate a 1D CLAP vector to a massive 2D STFT matrix. To combine them, I've wired Genesis to use **Feature-wise Linear Modulation (FiLM)**.



1. **Feature Extraction:** Genesis processes the STFT matrix through a convolutional encoder to extract structural hidden features, denoted as $H$.
2. **CLAP Translation:** Concurrently, Fabian's CLAP vector ($c_{scene}$) passes through a Multi-Layer Perceptron (MLP). I'm having it calculate two conditioning parameters:
   * Scaling factor: $\gamma(c)$
   * Shifting factor: $\beta(c)$
3. **Modulation:** Genesis then mathematically injects the semantic concept into the audio features using the FiLM equation:
   $$H' = \gamma(c) \cdot H + \beta(c)$$

The CLAP vector acts as a dynamic filter. It mathematically scales and shifts the STFT features to align with Fabian's target. The modulated features ($H'$) are then passed straight to the DDSP Decoder.

### 4. Raw Outputs
The DDSP Decoder translates $H'$ into literal, frame-by-frame parameters for two distinct DSP components.

1. **Impulse Response Generation:** Genesis outputs the exact frequency decay curves and delay times to compile a 32-bit float array Impulse Response ($IR(t)$). I load this into her Convolver plugin, which mathematically applies the acoustic space to the live audio ($x(t)$) via the convolution theorem:
   $$y(t) = (x * IR)(t) = \int_{-\infty}^{\infty} x(\tau) IR(t - \tau) d\tau$$
2. **Procedural Comfort Noise Generation (CNG):** If Fabian's vector implies background ambiance, Genesis does not output an audio file (which would loop and sound obvious). Instead, she outputs continuous mathematical seeds—LFO rates, bandpass cutoffs, and pink-noise amplitudes. This synthesizes infinite, non-repeating room tone beneath the primary signal.

### 5. Training Data and Methodology
Because Genesis is translating abstract vectors into hard physics via FiLM, her weights will be completely random at first. She will output screeching noise until I train her.

**The Datasets I'll Use:**
* **LAION-Audio-630K & AudioCaps:** I will use the text-to-audio pairs to establish the CLAP vectors.
* **MIT Acoustical Reverberation Scene Statistics (or OpenAir):** I will pull a massive corpus of real-world Impulse Responses from these libraries.
* **VCTK / LJSpeech:** I'll use these as my foundation of high-fidelity, completely dry studio vocals.

**How I Will Train Her (The Synthetic Pairing Loop):**
Genesis's training data must be synthetically constructed by me, because natural datasets do not contain the isolated variables she needs to learn. Here is exactly how I'll build her training ground:

1. **The Foundation:** I'll take a completely dry vocal ($V_{dry}$) from VCTK.
2. **The Target Environment:** I will mathematically convolve that dry vocal with a real-world IR ($IR_{real}$) to create a wet, reverberant vocal ($V_{wet}$).
3. **The Semantic Vector:** I'll pass that $V_{wet}$ audio through a frozen CLAP Audio Encoder to extract the mathematical concept of that room, giving me the target scene vector ($c_{scene}$).
4. **The Test:** I will hand Genesis the dry audio ($V_{dry}$) and the concept vector ($c_{scene}$), and command her to predict the exact DDSP parameters needed to generate $IR_{real}$.

**The Math I'll Use for Her Loss Function:**
I will penalize her by comparing the spectrogram of her generated audio ($\hat{S}$) against the true audio ($S$). I have to do this across multiple FFT sizes to ensure she gets both the transient micro-timing (the initial slapback echo) and the long-term frequency curves (the reverb tail) exactly right. 

I will use a **Multi-Scale Spectral Loss ($L_{MSS}$)** function:



$$L_{MSS} = \sum_{i} \left( ||S_i - \hat{S}_i||_1 + \alpha ||\log S_i - \log \hat{S}_i||_1 \right)$$

By backpropagating this error, the random weights in her FiLM layers will quickly organize into a flawless mathematical understanding of physical acoustic space. Because I am training her to output DSP parameters instead of drawing raw waveforms, this training will only take me a few days on my consumer GPU, rather than months on a server farm.