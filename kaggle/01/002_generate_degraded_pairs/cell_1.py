# ## Phase 2: Training Data — Degradation Pipeline (v2)
#
# **Storage-efficient approach**: Store paths to pristine audio in a JSONL file,
# not the audio itself. Each pair saves only the degraded WAV + params.
#
# **Two-pass pipeline**:
# - Pass 1: Scan datasets, pick clips (VCTK max 150/speaker, LJSpeech 4000, DAPS all),
#   write `chosen_pristine.jsonl`
# - Pass 2: For each entry, load pristine → extract 67D metrics → assign cluster_id_pristine
#   → degrade → extract 67D metrics → assign cluster_id_degraded → save degraded WAV + params
#
# **Output structure**:
# ```
# /kaggle/working/ursula_dataset/
# ├── metadata.json
# ├── chosen_pristine.jsonl
# ├── pairs/
# │   ├── 00000001/
# │   │   ├── 00000001.wav    # degraded only
# │   │   └── params.json     # degradation params + both cluster_ids
# │   └── ...
# ├── degradation_params.json
# └── checkpoint.json
# ```
