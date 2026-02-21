# Agent 1: Fabian (The Orchestrator & Multi-Target Router)

## Part 1: Overview
Fabian is the command center and semantic translator of the Fa-Ur-Ge pipeline. He is the only agent in the system that understands human language. 

Standard audio models require massive compute to parse English text alongside audio math. Fabian isolates that semantic workload. When a user asks to "make me sound like a 1920s radio host in an empty hallway," Fabian translates that text into a **Split Destination** based on the physical reality of audio engineering:
1. **The Source (For Ursula):** He calculates the exact physical metrics (Long-Term Average Spectrum, LUFS, Crest Factor) of a 1920s radio mic.
2. **The Scene (For Genesis):** He calculates the abstract latent concept (CLAP vector) of an empty hallway.

He places these distinct mathematical flags on the field, dictates who needs to wake up, and immediately unloads from VRAM.

---

## Part 2: Engineering Specification

### 1. Core Architecture
Fabian's brain is built on a **Contrastive Language-Audio Pretraining (CLAP)** text encoder. This frozen core natively understands the abstract geometry of the audio universe. 



Attached to this core are two custom, lightweight neural heads:
* **The Routing Head:** A binary classifier that decides which downstream agents are required.
* **The Physical Regression Head:** A multi-target network that maps the semantic text embedding into literal, physical audio metrics for Ursula to chase.

### 2. The Inputs
Fabian operates on two distinct data streams.

* **Input A: The Semantic Prompt (Text)**
  * **Source:** User CLI or GUI prompt.
  * **Structure:** Raw String.
* **Input B: The Baseline State (LTAS / LUFS / CLAP)**
  * **Source:** The Shadow Space Audio Analyzer.
  * **Function:** Fabian looks at the current physical and semantic metrics of the raw microphone audio to calculate the routing logic.

### 3. Processing Mechanics & The Split Handoff
Fabian tokenizes the text and passes it through the CLAP encoder to get the base text embedding ($e_{text}$). He then bifurcates the math:

1. **Scene Target (Genesis):** The text embedding inherently acts as the CLAP vector.
   $$c_{scene} = e_{text} \in \mathbb{R}^{512}$$
2. **Source Targets (Ursula):** The text embedding passes through the Physical Regression Head to output explicit DSP metrics:
   * Target LTAS (Frequency curve): $T_{LTAS} \in \mathbb{R}^{64}$
   * Target LUFS (Loudness): $T_{LUFS} \in \mathbb{R}$
   * Target ZCR/Crest Factor (Dynamics/Harshness): $T_{Dyn} \in \mathbb{R}^2$

### 4. Raw Outputs
Fabian emits these variables as a highly structured dictionary:

1. **Target Matrices:** * `{"scene_clap": [0.8, -0.2...], "source_ltas": [...], "source_lufs": -14.0, "source_lra": 4.5}`
2. **Routing Flags:** * `{"requires_ursula": True, "requires_genesis": True, "requires_cng": False}`

### 5. Training Data and Methodology
Because the core CLAP model was pre-trained by researchers, his semantic understanding of the audio universe is already baked in. I will leave the core weights completely frozen. 

However, I have to train his **Physical Regression Head** and his **Routing Head** from scratch so he learns how to map English adjectives to exact LUFS and LTAS values.

**The Datasets I'll Use:**
* **LAION-Audio-630K & AudioCaps:** I'll use the text descriptions from these massive datasets.
* **My Custom Physical Labels:** I'll write a Python script to iterate through the audio files in those datasets, calculating their literal LTAS, LUFS, and LRA.

**How I Will Train Him (The Translation Loop):**
1. **The Setup:** I will freeze the CLAP text encoder. I'll attach my custom, untrained Multi-Layer Perceptrons (MLPs) to its output.
2. **The Feed:** I will pass text descriptions (e.g., "A loud, booming radio voice") through Fabian. 
3. **The Target:** I will ask his untrained Physical Regression Head to guess the LTAS curve and the LUFS value for that text.
4. **The Loss Function:** I will penalize him by comparing his guessed physical metrics against the *actual* metrics I extracted from the corresponding audio file. I will use a simple Mean Squared Error (MSE) loss for this:
   $$L_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   *(Where $y_i$ is the true LTAS/LUFS of the audio, and $\hat{y}_i$ is Fabian's predicted LTAS/LUFS).*

Because I'm only training a few linear layers on top of a frozen model, I can teach Fabian to perfectly map English adjectives to hard audio physics in under an hour on my GPU.