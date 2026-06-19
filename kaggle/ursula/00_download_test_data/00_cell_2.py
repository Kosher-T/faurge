import re

# ─── DAPS Utilities ──────────────────────────────────────────────────

def parse_daps_filename(filename: str):
    """
    Parse DAPS filename to extract speaker and script ID.
    Example: 'f1_script1_cleanraw.wav' -> 'f1', 'script1'
    """
    stem = Path(filename).stem
    # Match speaker ID (f# or m#)
    match = re.match(r"^([fm]\d+)_(.+)$", stem)
    if not match:
        return None, None
    
    speaker = match.group(1)
    rest = match.group(2)
    
    # Strip any environment-related suffixes to isolate script ID
    suffixes_to_strip = [
        "_cleanraw", "_clean",
        "_iphone_livingroom1", "_iphone_livingroom",
        "_ipad_office2", "_ipad_office",
        "_ipad_confroom2", "_ipad_confroom",
        "_ipad_balcony1", "_ipad_balcony"
    ]
    script = rest
    for suffix in suffixes_to_strip:
        if script.endswith(suffix):
            script = script[:-len(suffix)]
            break
            
    return speaker, script


def discover_daps_speakers_and_scripts(cleanraw_dir: Path):
    """
    Scan cleanraw directory and map speaker -> list of unique script IDs.
    """
    speaker_to_scripts = defaultdict(list)
    if not cleanraw_dir.exists():
        print(f"Warning: DAPS cleanraw directory {cleanraw_dir} does not exist.")
        return speaker_to_scripts
        
    for p in sorted(cleanraw_dir.glob("*.wav")):
        speaker, script = parse_daps_filename(p.name)
        if speaker and script:
            speaker_to_scripts[speaker].append(script)
            
    # Deduplicate and sort scripts for each speaker
    for speaker in speaker_to_scripts:
        speaker_to_scripts[speaker] = sorted(list(set(speaker_to_scripts[speaker])))
        
    return speaker_to_scripts


def find_daps_file(env_dir: Path, speaker: str, script: str) -> Path | None:
    """
    Find the file matching the speaker and script prefix within an environment folder.
    """
    if not env_dir.exists():
        return None
    prefix = f"{speaker}_{script}_"
    for p in env_dir.glob("*.wav"):
        if p.name.startswith(prefix):
            return p
    return None


def copy_daps_subset(base_dir: Path, output_dir: Path, environments: list[str], scripts_per_speaker: int = 2):
    """
    Randomly select scripts_per_speaker for each speaker, and copy the corresponding files
    across all environment folders to the output directory.
    """
    cleanraw_dir = base_dir / "cleanraw"
    speakers_scripts = discover_daps_speakers_and_scripts(cleanraw_dir)
    
    if not speakers_scripts:
        print("No DAPS speakers found.")
        return
        
    print(f"Found {len(speakers_scripts)} DAPS speakers.")
    total_copied = 0
    missing_files = []
    
    for idx, (speaker, scripts) in enumerate(sorted(speakers_scripts.items()), 1):
        if len(scripts) < scripts_per_speaker:
            selected_scripts = scripts
        else:
            selected_scripts = random.sample(scripts, scripts_per_speaker)
            
        print(f"  [{idx}/{len(speakers_scripts)}] Speaker {speaker}: selecting scripts {selected_scripts}")
        
        for script in selected_scripts:
            # Copy across all environments
            for env in environments:
                env_dir = base_dir / env
                src_file = find_daps_file(env_dir, speaker, script)
                
                if src_file and src_file.exists():
                    dest_file = output_dir / "daps" / env / src_file.name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                    total_copied += 1
                else:
                    missing_files.append(f"{speaker}_{script} in {env}")
                    
    print(f"Successfully copied {total_copied} DAPS files.")
    if missing_files:
        print(f"Warning: {len(missing_files)} expected files were not found across environments.")

# ─── VCTK Utilities ──────────────────────────────────────────────────

def discover_vctk_speakers(wav48_dir: Path, min_clips: int = 50):
    """
    Discover all speaker directories in VCTK that contain at least min_clips.
    """
    speakers = []
    if not wav48_dir.exists():
        print(f"Warning: VCTK base directory {wav48_dir} does not exist.")
        return speakers
        
    for p in sorted(wav48_dir.iterdir()):
        if p.is_dir():
            wav_files = list(p.glob("*.wav"))
            if len(wav_files) >= min_clips:
                speakers.append((p.name, p))
            else:
                print(f"Skipping VCTK speaker {p.name} (has {len(wav_files)} clips, need >= {min_clips})")
    return speakers


def copy_vctk_subset(wav48_dir: Path, output_dir: Path, n_speakers: int = 50, clips_per_speaker: int = 50):
    """
    Select n_speakers at random, and copy clips_per_speaker random files from each.
    """
    valid_speakers = discover_vctk_speakers(wav48_dir, clips_per_speaker)
    if not valid_speakers:
        print("No valid VCTK speakers found.")
        return
        
    print(f"Found {len(valid_speakers)} valid VCTK speakers with >= {clips_per_speaker} clips.")
    selected_speakers = random.sample(valid_speakers, min(n_speakers, len(valid_speakers)))
    print(f"Selected {len(selected_speakers)} VCTK speakers for subsetting.")
    
    total_copied = 0
    for idx, (speaker_name, speaker_path) in enumerate(selected_speakers, 1):
        wav_files = list(speaker_path.glob("*.wav"))
        selected_clips = random.sample(wav_files, clips_per_speaker)
        
        sp_out_dir = output_dir / "vctk" / speaker_name
        sp_out_dir.mkdir(parents=True, exist_ok=True)
        
        for clip in selected_clips:
            dest = sp_out_dir / clip.name
            shutil.copy2(clip, dest)
            total_copied += 1
            
        if idx % 10 == 0 or idx == len(selected_speakers):
            print(f"  Processed {idx}/{len(selected_speakers)} VCTK speakers...")
            
    print(f"Successfully copied {total_copied} VCTK files.")

# ─── Archive Utilities ────────────────────────────────────────────────

def zip_output_dir(source_dir: Path, output_zip: Path):
    """
    Zip the contents of source_dir to output_zip.
    """
    print(f"Creating zip file at {output_zip}...")
    t0 = time.time()
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)
    elapsed = time.time() - t0
    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Zip completed in {elapsed:.1f}s — Size: {zip_size_mb:.1f} MB")
