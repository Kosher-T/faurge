# %% [markdown]
# ## UrsulaDSPEnv — Gymnasium Environment
#
# The core RL environment for training Ursula's DSP policy.
# Each episode: load a degraded clip, apply the 7-plugin cascade from the
# policy's 227D action, compute metrics, and return reward based on MSE
# improvement toward the reference metrics.

class UrsulaDSPEnv(gym.Env):
    """
    Gymnasium environment for training Ursula's DSP restoration policy.

    Observation (143D):
        [M_current (67), M_reference (67), cluster_onehot (9)]

    Action (227D):
        Normalized params in [-1, 1] → decoded to real plugin ranges

    Reward:
        -soft_clamp(MSE(M_current, M_reference) - identity_floor, k)

    Episode terminates when:
        - MSE < identity_floor (success), or
        - steps >= max_steps
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pairs_data: Path = None,
        metrics_data: Path = None,
        cluster_data: Path = None,
        max_steps: int = MAX_STEPS,
        soft_clamp_k: float = 10.0,
        mode: str = "train",  # "train" or "eval"
    ):
        super().__init__()

        self.pairs_data = pairs_data or PAIRS_DATA
        self.metrics_data = metrics_data or METRICS_DATA
        self.cluster_data = cluster_data or CLUSTER_DATA
        self.max_steps = max_steps
        self.soft_clamp_k = soft_clamp_k
        self.mode = mode

        # ── Load metadata ──
        self._load_metadata()

        # ── Load paths.csv ──
        self._load_paths()

        # ── Load identity floors ──
        self._load_identity_floors()

        # ── Spaces ──
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(INPUT_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OUTPUT_DIM,), dtype=np.float32,
        )

        # ── Internal state ──
        self._current_audio = None
        self._current_metrics = None
        self._reference_metrics = None
        self._cluster_onehot = None
        self._cluster_id = None
        self._pair_info = None
        self._step_count = 0
        self._current_mse = None

        print(f"UrsulaDSPEnv initialized: {len(self._pairs)} pairs, mode={mode}")

    def _load_metadata(self):
        """Load metadata.json from metrics dataset."""
        meta_path = self.metrics_data / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}
            print(f"  [WARN] No metadata.json found at {meta_path}")

    def _load_paths(self):
        """Load paths.csv — maps pair_id to degraded/reference paths and cluster_id."""
        csv_path = self.metrics_data / 'paths.csv'
        self._pairs = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._pairs.append(row)
        print(f"  Loaded {len(self._pairs)} pairs from {csv_path}")

    def _load_identity_floors(self):
        """Load identity_floors.json for reward clamping."""
        floors_path = self.cluster_data / 'identity_floors.json'
        if floors_path.exists():
            with open(floors_path) as f:
                raw = json.load(f)
            # Convert to floats
            self._identity_floors = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    self._identity_floors[k] = {kk: float(vv) for kk, vv in v.items()}
                else:
                    self._identity_floors[k] = float(v)
        else:
            self._identity_floors = {}
            print(f"  [WARN] No identity_floors.json found at {floors_path}")

        # Default floor for unknown clusters
        self._default_floor = 0.05

    def _get_floor_for_cluster(self, cluster_id: int) -> float:
        """Get identity floor for a given cluster."""
        key = f"cluster_{cluster_id}"
        if key in self._identity_floors:
            return self._identity_floors[key]
        return self._default_floor

    def _build_onehot(self, cluster_id: int) -> np.ndarray:
        """Build (N_CLUSTERS_ONEHOT,) one-hot vector."""
        onehot = np.zeros(N_CLUSTERS_ONEHOT, dtype=np.float32)
        if 0 <= cluster_id < N_CLUSTERS:
            onehot[cluster_id] = 1.0
        else:
            onehot[N_CLUSTERS] = 1.0  # unknown
        return onehot

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio file, trim/pad to CLIP_SAMPLES."""
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        # Trim or pad to CLIP_SAMPLES
        if len(audio) > CLIP_SAMPLES:
            audio = audio[:CLIP_SAMPLES]
        elif len(audio) < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        return audio

    def _load_metrics(self, pair_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load .pt file → (M_degraded, M_reference), each (67,)."""
        pt_path = self.metrics_data / 'pairs' / f'{pair_id}.pt'
        tensor = torch.load(pt_path, map_location='cpu', weights_only=True)
        return tensor[0].numpy(), tensor[1].numpy()

    def reset(self, seed=None, options=None):
        """Reset environment with a new pair."""
        super().reset(seed=seed)

        # Sample a random pair
        idx = self.np_random.integers(len(self._pairs))
        self._pair_info = self._pairs[idx]
        pair_id = self._pair_info['pair_id']

        # Load audio
        degraded_path = self._pair_info['degraded_path']
        self._current_audio = self._load_audio(degraded_path)

        # Load metrics
        self._current_metrics, self._reference_metrics = self._load_metrics(pair_id)

        # Cluster one-hot
        self._cluster_id = int(self._pair_info['cluster_id_reference'])
        self._cluster_onehot = self._build_onehot(self._cluster_id)

        # Reset step counter
        self._step_count = 0
        self._current_mse = float(np.mean((self._current_metrics - self._reference_metrics) ** 2))

        # Build observation
        obs = np.concatenate([
            self._current_metrics,
            self._reference_metrics,
            self._cluster_onehot,
        ]).astype(np.float32)

        info = {
            "pair_id": pair_id,
            "cluster_id": self._cluster_id,
            "initial_mse": self._current_mse,
            "identity_floor": self._get_floor_for_cluster(self._cluster_id),
        }

        return obs, info

    def step(self, action):
        """Apply action (227D) to current audio, return new observation + reward."""
        self._step_count += 1

        # Decode action → plugin config dicts
        plugin_dicts = decode_action(action)

        # Apply 7-plugin cascade
        try:
            processed_audio = apply_plugins(self._current_audio, SR, plugin_dicts)
        except Exception as e:
            # If plugins fail, return penalty and keep current state
            print(f"  [WARN] Plugin error: {e}")
            obs = np.concatenate([
                self._current_metrics,
                self._reference_metrics,
                self._cluster_onehot,
            ]).astype(np.float32)
            return obs, -1.0, False, False, {"error": str(e)}

        # Compute metrics of processed audio
        try:
            m_result = extract_metrics_67d(processed_audio)
        except Exception as e:
            print(f"  [WARN] Metrics error: {e}")
            obs = np.concatenate([
                self._current_metrics,
                self._reference_metrics,
                self._cluster_onehot,
            ]).astype(np.float32)
            return obs, -1.0, False, False, {"error": str(e)}

        # Compute MSE
        mse = float(np.mean((m_result - self._reference_metrics) ** 2))

        # Reward: -soft_clamp(MSE - identity_floor, k)
        floor = self._get_floor_for_cluster(self._cluster_id)
        reward = soft_clamp(mse - floor, k=self.soft_clamp_k)

        # Update state
        self._current_audio = processed_audio
        self._current_metrics = m_result
        self._current_mse = mse

        # Build observation
        obs = np.concatenate([
            self._current_metrics,
            self._reference_metrics,
            self._cluster_onehot,
        ]).astype(np.float32)

        # Termination conditions
        terminated = mse < floor  # success: below identity floor
        truncated = self._step_count >= self.max_steps

        info = {
            "mse": mse,
            "identity_floor": floor,
            "step": self._step_count,
            "delta_mse": self._current_mse - mse,
        }

        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Return mask indicating which action dims are 'active' (>0.01 absolute)."""
        return np.ones(OUTPUT_DIM, dtype=np.float32)

    def close(self):
        pass


print("UrsulaDSPEnv class defined")
