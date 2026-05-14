# Migration Note: Genesis Data Format

These notebooks were written for the old Genesis pipeline where the model
received STFT + text prompts and output IR parameters.

The new Genesis (as of the Fabian-as-orchestrator update) receives
(E_dsp, E_ref) — pairs of CLAP embeddings — and outputs an IR.

## Required Notebook Rewrite

`01_acquire_and_augment.ipynb` currently outputs:
- `source_audio` (int16) — degraded vocal
- `target_audio` (int16) — reference vocal convolved with target IR
- `target_clap` (float32) — CLAP of target IR + noise

**New format should output:**
- `dsp_audio` (int16) — audio after DSP simulation (mimics Ursula's output)
- `ideal_audio` (int16) — reference convolved with target IR (ground truth for loss)
- `dsp_clap` (float32 512D) — CLAP embedding of dsp_audio
- `ideal_clap` (float32 512D) — CLAP embedding of ideal_audio
- `target_ir` (int16) — ground truth IR waveform

See `docs/genesis.md` Section 4 for the full training data specification.
