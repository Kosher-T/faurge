import json
import logging
import os
import tempfile
from pathlib import Path

import pytest
import jsonschema

from core import defaults

def test_defaults_constants_exist():
    """Assert every expected constant is importable from core.defaults"""
    assert hasattr(defaults, "GPU_TOTAL_VRAM_MB")
    assert hasattr(defaults, "AUDIO_SAMPLE_RATE")
    assert defaults.GPU_TOTAL_VRAM_MB == 4096
    assert defaults.AUDIO_SAMPLE_RATE == 48000

def test_settings_loads_defaults(monkeypatch):
    """Test standard loading without env vars"""
    # clear env vars
    monkeypatch.delenv("FAURGE_GPU_TOTAL_VRAM_MB", raising=False)
    
    # Reload settings module to re-evaluate constants
    import importlib
    import core.settings
    importlib.reload(core.settings)
    
    assert core.settings.GPU_TOTAL_VRAM_MB == defaults.GPU_TOTAL_VRAM_MB
    assert core.settings.MAX_LATENCY_MS == defaults.MAX_LATENCY_MS

def test_settings_env_override(monkeypatch):
    """Test that environment variables successfully override defaults"""
    monkeypatch.setenv("FAURGE_GPU_TOTAL_VRAM_MB", "8192")
    monkeypatch.setenv("FAURGE_BYPASS_ON_VRAM_SPIKE", "false")
    monkeypatch.setenv("FAURGE_AUDIO_SAMPLE_RATE", "96000")
    
    import importlib
    import core.settings
    importlib.reload(core.settings)
    
    assert core.settings.GPU_TOTAL_VRAM_MB == 8192
    assert core.settings.BYPASS_ON_VRAM_SPIKE is False
    assert core.settings.AUDIO_SAMPLE_RATE == 96000

def test_schema_rejects_bad_config(monkeypatch):
    """Test schema validation catches bad overrides"""
    monkeypatch.setenv("FAURGE_MAX_LATENCY_MS", "999") # well above allowed 100
    
    import importlib
    import core.settings
    with pytest.raises(RuntimeError, match="Faurge Configuration Error"):
        importlib.reload(core.settings)

def test_logger_creates_handlers():
    from core.logging import get_logger
    from logging.handlers import RotatingFileHandler
    
    logger = get_logger("test_faurge")
    
    assert len(logger.handlers) == 2
    types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in types
    assert RotatingFileHandler in types

def test_logger_writes_json_lines():
    from core.logging import get_logger, JsonLinesFormatter
    logger = get_logger("test_json")
    
    # Find the rotating file handler and get its file path
    from logging.handlers import RotatingFileHandler
    fh = next(h for h in logger.handlers if isinstance(h, RotatingFileHandler))
    log_file = fh.baseFilename
    
    # Empty it
    open(log_file, 'w').close()
    
    # Write a log
    logger.info("Test JSON log entry")
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        
        log_obj = json.loads(lines[0])
        assert log_obj["level"] == "INFO"
        assert log_obj["logger"] == "test_json"
        assert log_obj["message"] == "Test JSON log entry"
        assert "timestamp" in log_obj
