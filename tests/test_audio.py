"""Tests for audio probing, normalization and WAV round-trips."""

from __future__ import annotations

import math
import wave

import pytest

from attune.core.audio import (
    AudioError,
    is_normalized,
    normalize_to_wav,
    probe,
    read_wav_mono,
    write_wav,
)


def test_write_and_read_wav_round_trip(tmp_path) -> None:
    path = write_wav(tmp_path / "a.wav", [0.0, 0.5, -0.5, 1.0], 16000)
    samples, rate = read_wav_mono(path)
    assert rate == 16000
    assert len(samples) == 4
    assert math.isclose(samples[1], 0.5, abs_tol=1e-4)
    assert math.isclose(samples[2], -0.5, abs_tol=1e-4)


def test_probe_reports_format(tmp_path) -> None:
    path = write_wav(tmp_path / "a.wav", [0.0] * 16000, 16000)
    info = probe(path)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert math.isclose(info.duration, 1.0, abs_tol=0.01)


def test_is_normalized_detects_target_format(tmp_path) -> None:
    good = write_wav(tmp_path / "good.wav", [0.0] * 100, 16000)
    assert is_normalized(good) is True

    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00\x00\x00" * 10)
    assert is_normalized(stereo) is False


def test_normalize_copies_already_normalized_files(tmp_path) -> None:
    source = write_wav(tmp_path / "a.wav", [0.0] * 10, 16000)
    target = tmp_path / "out" / "b.wav"
    result = normalize_to_wav(source, target)
    assert result == str(target)
    assert target.exists()


def test_normalize_reports_missing_ffmpeg(tmp_path, monkeypatch) -> None:
    import attune.core.audio as audio_module

    monkeypatch.setattr(audio_module.shutil, "which", lambda name: None)
    source = tmp_path / "a.mp3"
    source.write_bytes(b"not really audio")
    with pytest.raises(AudioError):
        normalize_to_wav(source, tmp_path / "out.wav")


def test_probe_missing_file_raises(tmp_path) -> None:
    with pytest.raises(AudioError):
        probe(tmp_path / "nope.wav")
