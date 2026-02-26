# Faurge Agentic Execution Flow (The Closed Loop)

This document outlines the execution pipeline of the Faurge tensor network, specifically the closed-loop reflection system managed by `core/bake_orchestrator.py`. The Orchestrator acts as the memory bank and mathematical integrator to ensure the AI agents remain completely stateless and adhere to the strict 4GB VRAM constraint.

---

## Phase 1: CPU Pre-Flight & The Intent Fallback
To protect the mathematical integrity of the latent space without using VRAM-heavy LLMs, all user text is sanitized upstream on the CPU.
1. **The Intercept:** The `PromptSanitizer` scans the user's raw string.
2. **Intent Fallback:** If the prompt is overly vague (e.g., "make it better", "fix my mic"), the sanitizer deletes the string and replaces it with a mathematically dense Anchor Prompt (e.g., "A clear, professional studio broadcast vocal, high fidelity, smooth EQ, flat frequency response, no background noise").
3. **Sanitization:** Typos are corrected, and heavy slang is mapped to concrete acoustic adjectives via hardcoded dictionaries.

## Phase 2: Round 1 (The Blueprint)
1. **State Fusion:**
   * The text prompt passes through the CLAP text encoder $\rightarrow e_{text} \in \mathbb{R}^{512}$
   * The raw microphone buffer passes through the CLAP audio encoder $\rightarrow e_{audio\_current} \in \mathbb{R}^{512}$
   * The Orchestrator concatenates them: `numpy.concatenate([e_text, e_audio_current], axis=-1)`
2. **Fabian (The Router):** Wakes up, receives the 1024D tensor, and calculates the absolute baseline targets ($T_0$).
   * *For Ursula:* $T_{LTAS}$ (Frequency), $T_{LUFS}$ (Loudness), $T_{Dyn}$ (Dynamics).
   * *For Genesis:* $c_{scene}$ (The raw 512D text embedding).
   * Fabian unloads from VRAM.
3. **Actuation (One-Shot Execution):**
   * **Ursula:** Receives $T_0$ targets and the raw audio's current metrics. She performs a **single forward pass** through her trained policy network to output absolute `dsp_state` parameters for the C++ EQ/Comp plugins. There is no live Reinforcement Learning or trial-and-error here. She unloads.
   * **Genesis:** Receives $c_{scene}$ and the High-Res STFT of the raw audio. She performs a **single forward pass** through her DDSP network to mathematically compile a custom `ir` to bridge the harmonic gap. She does not evaluate her own work. She unloads.
4. **The Shadow Bake:** The Orchestrator applies the `dsp_state` and `ir` to the buffer to generate the *Proposed Audio*.

## Phase 3: The Reflection Loop (Accumulation)
1. **Evaluation:** Fabian wakes back up. The *Proposed Audio* is encoded to a new $e_{audio\_current}$. Fabian calculates the Cosine Similarity ($S_C$) between $e_{text}$ and the new $e_{audio\_current}$.
2. **Decision:** If $S_C \ge 0.85$, the loop terminates successfully.
3. **The Delta Generation (Fail State):** If the audio still misses the mark (e.g., residual distortion), Fabian's neural network sees the remaining gap and outputs **Deltas** (refinements) instead of absolute targets: $\Delta T_{LTAS}$, $\Delta T_{LUFS}$, $\Delta T_{Dyn}$, and $\Delta c_{scene}$. Fabian unloads.
4. **The Integrator Math:** To prevent oscillation, the Orchestrator accumulates the targets:
   $$T_{new} = T_{n-1} + \Delta T_n$$
5. **Re-Actuation (One-Shot Refinement):**
   * Ursula receives the accumulated targets ($T_{new}$) and performs another single forward pass to output completely *new* `dsp_state` parameters.
   * Genesis receives the accumulated CLAP target and performs another single forward pass to mathematically compile a brand *new* `ir` from scratch.
   * (Loop repeats until $S_C$ passes).

## Phase 4: Live Execution
1. **Overwrite:** The Orchestrator pushes the finalized `dsp_state` to the active PipeWire plugins, replacing the old parameters.
2. **Cross-fade:** `core/state_manager.py` executes a 10ms cross-fade inside the PipeWire convolver, swapping the old IR for the newly finalized IR.

### 6. Cloud Training Methodology (State-Aware Regression)

To function as the controller in Faurge's closed-loop system, Fabian's Physical Regression Head is not trained to predict absolute targets. It is trained to predict the **Delta ($\Delta$)**—the exact mathematical difference between the current audio state and the desired audio state.

This training takes place in the cloud (Phase 3) using Kaggle's GPU infrastructure.

#### Step 1: Dataset Synthetic Degradation
We cannot use the datasets exactly as they are. We must simulate the concept of a "cheap microphone" or a "failed reflection pass."
1. **The Pristine Source:** We take a high-quality audio file and its text description (e.g., "A clean, booming radio voice") from the LAION/AudioCaps datasets.
2. **The Degradation Pass:** We aggressively process a copy of this file using standard Python DSP libraries (e.g., `pedalboard`). We roll off the bass, compress it terribly, and inject white noise to simulate a bad baseline state.

#### Step 2: Ground Truth Delta Extraction
Instead of predicting the absolute metrics of the pristine file, Fabian must predict the distance between the degraded file and the pristine file.
1. We measure the physical metrics ($T_{LTAS}$, $T_{LUFS}$, $T_{Dyn}$, $T_{Purity}$) of **both** the Pristine file and the Degraded file.
2. The ground truth labels for the neural network are the literal differences:
   $$\Delta T_{target} = T_{pristine} - T_{degraded}$$

#### Step 3: State Fusion (The Forward Pass)
We simulate the Orchestrator's exact state-fusion logic.
1. The pristine text description passes through the frozen CLAP text encoder $\rightarrow e_{text}$.
2. The *Degraded* audio passes through the frozen CLAP audio encoder $\rightarrow e_{audio\_degraded}$.
3. The tensors are concatenated into the 1024-dimensional input matrix: $[e_{text}, e_{audio\_degraded}]$.

#### Step 4: Backpropagation & Loss
1. **Inference:** Fabian's untrained Physical Regression Head receives the 1024D input and attempts to guess the required correction: $\Delta T_{predicted}$.
2. **MSE Loss:** We penalize Fabian based on how far his predicted Delta is from the true Delta using Mean Squared Error:
   $$L_{MSE} = \frac{1}{N} \sum_{i=1}^N (\Delta T_{target} - \Delta T_{predicted})^2$$
3. **Routing Head Loss:** Simultaneously, the Routing Head evaluates the $T_{Purity}$ of the degraded file. If the Purity is below a hardcoded threshold (indicating crackle/distortion), the target flag for `requires_genesis` is set to $1$. We use Binary Cross-Entropy (BCE) loss to train this binary classification flag.

**Why this works:** By training Fabian to predict the mathematical gap ($\Delta T$) between a degraded state and a pristine state, he natively handles both Round 1 and Round 2 of the live Orchestrator loop. In Round 1, the "degraded state" is the user's raw mic. In Round 2, the "degraded state" is the *Proposed Audio* that just missed the mark. His network simply outputs the required offset to reach the goal.