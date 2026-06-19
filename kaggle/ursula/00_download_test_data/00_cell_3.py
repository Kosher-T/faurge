t_total = time.time()

# ── Clean/Prepare Output Directory ──
if OUTPUT_DIR.exists():
    print(f"Cleaning existing output directory {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Subset and Copy DAPS ──
print("\n═══ Step 1: Subsetting and Copying DAPS ═══")
t_start_daps = time.time()
copy_daps_subset(
    base_dir=DAPS_BASE_DIR,
    output_dir=OUTPUT_DIR,
    environments=DAPS_ENVIRONMENTS,
    scripts_per_speaker=2
)
print(f"DAPS processing finished in {time.time() - t_start_daps:.1f}s")

# ── Step 2: Subset and Copy VCTK ──
print("\n═══ Step 2: Subsetting and Copying VCTK ═══")
t_start_vctk = time.time()
copy_vctk_subset(
    wav48_dir=VCTK_BASE_DIR,
    output_dir=OUTPUT_DIR,
    n_speakers=50,
    clips_per_speaker=50
)
print(f"VCTK processing finished in {time.time() - t_start_vctk:.1f}s")

# ── Step 3: Package Output Directory into Zip ──
print("\n═══ Step 3: Packaging Dataset Subset into Zip ═══")
if OUTPUT_ZIP.exists():
    OUTPUT_ZIP.unlink()
zip_output_dir(OUTPUT_DIR, OUTPUT_ZIP)

# ── Step 4: Cleanup Temporary Files ──
print("\n═══ Step 4: Cleaning up temporary folder ═══")
shutil.rmtree(OUTPUT_DIR)
print(f"Removed temporary directory: {OUTPUT_DIR}")

# ── Summary ──
elapsed = time.time() - t_total
print(f"\n{'='*60}")
print(f"  DATASET SUBSETTING COMPLETE — {elapsed/60:.1f} min")
print(f"  Downloadable Archive: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / (1024 * 1024):.1f} MB)")
print(f"{'='*60}")
