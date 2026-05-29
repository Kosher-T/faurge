# %% [markdown]
# # Phase 1: Cluster Analysis & Identity Floor
#
# Clusters speakers by voice similarity using 67D physical metrics (LTAS + LUFS + Dynamic Range).
# Outputs cluster assignments, centroids with thresholds, and identity floors for Ursula's RL training.
#
# **Inputs:** Pristine audio from VCTK, LJSpeech, DAPS
# **Outputs:** `clusters.json`, `cluster_centroids.json`, `identity_floors.json`
