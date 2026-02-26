# Faurge Agentic Execution Flow (The Closed Loop)

This document outlines the execution pipeline of the Faurge tensor network, specifically the closed-loop reflection system managed by `core/bake_orchestrator.py`. The Orchestrator acts as the memory bank and mathematical integrator to ensure the AI agents remain completely stateless and adhere to the strict 4GB VRAM constraint.

---

## Phase 1: CPU Pre-Flight & The Intent Fallback
To protect the mathematical integrity of the latent space without using VRAM-heavy LLMs, all user text is sanitized upstream on the CPU.
1. **The Intercept:** The `PromptSanitizer` scans the user's raw string.
2. **Intent Fallback:** If the prompt is overly vague (e.g., "make it better", "fix my mic"), the sanitizer deletes the string and replaces it with a mathematically dense Anchor Prompt (e.g., "A clear, professional studio broadcast vocal, high fidelity, smooth EQ, flat frequency response, no background noise").
3. **Sanitization:** Typos are corrected, and heavy slang is mapped to concrete acoustic adjectives via hardcoded dictionaries.

## Phase 2: Round 1 (The Blueprint)
1. **State Fusion:** * The text prompt passes through the CLAP text encoder $\rightarrow e_{text} \in \mathbb{R}^{512}$
   * The raw microphone buffer passes through the CLAP audio encoder $\rightarrow e_{audio\_current} \in \mathbb{R}^{512}$
   * The Orchestrator concatenates them: `numpy.concatenate([e_text, e_audio_current], axis=-1)`
2. **Fabian (The Router):** Wakes up, receives the 1024D tensor, and calculates the absolute baseline targets ($T_0$).
   * *For Ursula:* $T_{LTAS}$ (Frequency), $T_{LUFS}$ (Loudness), $T_{Dyn}$ (Dynamics).
   * *For Genesis:* $c_{scene}$ (The raw 512D text embedding).
   * Fabian unloads from VRAM.
3. **Actuation:** * **Ursula:** Receives $T_0$ targets and the raw audio's current metrics. Uses RL to output absolute `dsp_state` parameters for the C++ EQ/Comp plugins.
   * **Genesis:** Receives $c_{scene}$ and the High-Res STFT of the raw audio. Compiles a custom `ir` to bridge the harmonic gap.
4. **The Shadow Bake:** The Orchestrator applies the `dsp_state` and `ir` to the buffer to generate the *Proposed Audio*.

## Phase 3: The Reflection Loop (Accumulation)
1. **Evaluation:** Fabian wakes back up. The *Proposed Audio* is encoded to a new $e_{audio\_current}$. Fabian calculates the Cosine Similarity ($S_C$) between $e_{text}$ and the new $e_{audio\_current}$.
2. **Decision:** If $S_C \ge 0.85$, the loop terminates successfully.
3. **The Delta Generation (Fail State):** If the audio still misses the mark (e.g., residual distortion), Fabian's neural network sees the remaining gap and outputs **Deltas** (refinements) instead of absolute targets: $\Delta T_{LTAS}$, $\Delta T_{LUFS}$, $\Delta T_{Dyn}$, and $\Delta c_{scene}$. Fabian unloads.
4. **The Integrator Math:** To prevent oscillation, the Orchestrator accumulates the targets:
   $$T_{new} = T_{n-1} + \Delta T_n$$
5. **Re-Actuation:** * Ursula receives the accumulated targets ($T_{new}$) and outputs completely *new* `dsp_state` parameters.
   * Genesis receives the accumulated CLAP target and mathematically compiles a brand *new* `ir` from scratch.
   * (Loop repeats until $S_C$ passes).

## Phase 4: Live Execution
1. **Overwrite:** The Orchestrator pushes the finalized `dsp_state` to the active PipeWire plugins, replacing the old parameters.
2. **Cross-fade:** `core/state_manager.py` executes a 10ms cross-fade inside the PipeWire convolver, swapping the old IR for the newly finalized IR.