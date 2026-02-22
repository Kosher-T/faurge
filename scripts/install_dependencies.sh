#!/bin/bash
# scripts/install_dependencies.sh
# Bypasses pip's strict dependency resolver by installing Faurge in structural layers.
# This protects the 4GB TensorFlow cache and forces legacy audio math to comply.

set -e # Exit immediately if a command fails

echo "🟢 Layer 1: Core AI & Tensor Cache (Using local cache if available)..."
pip install tensorflow>=2.15.0 tf-keras>=2.15.0 transformers>=4.30.0

echo "🟢 Layer 2: Modern Math Anchors (Preventing 2020 rollbacks)..."
pip install numpy>=1.26.0 scipy>=1.11.0 numba>=0.59.0 resampy>=0.4.0 llvmlite>=0.42.0

echo "🟢 Layer 3: Audio Extraction & System Stack..."
pip install librosa>=0.10.0 soundfile>=0.12.1 pyloudnorm>=0.1.1 python-osc>=1.8.3 gymnasium>=0.29.0 pynvml>=11.5.0

echo "🟢 Layer 4: DDSP Prerequisites..."
pip install tensorflow-probability gin-config

echo "🟢 Layer 5: Forcing DDSP (Bypassing Dependency Checks)..."
pip install --no-deps ddsp>=1.2.0

echo "🟢 Layer 6: Development & Testing Tools..."
pip install -r requirements-dev.txt

echo "✅ Faurge Environment Successfully Locked."