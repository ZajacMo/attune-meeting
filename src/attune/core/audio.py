"""Audio helpers: probing, normalization to 16 kHz mono WAV, and slicing."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


class AudioError(RuntimeError):
    """Raised when an audio file cannot be read or converted."""


@dataclass(frozen=True)
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    duration: float

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration": round(self.duration, 3),
        }


def probe(path: str | Path) -> AudioInfo:
    """Inspect an audio file without converting it."""
    target = Path(path)
    if not target.exists():
        raise AudioError(f"Audio file not found: {target}")

    if target.suffix.lower() == ".wav":
        try:
            with wave.open(str(target), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                channels = handle.getnchannels()
                duration = frames / float(rate) if rate else 0.0
                return AudioInfo(str(target), rate, channels, duration)
        except wave.Error as error:  # pragma: no cover - depends on file content
            raise AudioError(f"Cannot read WAV file {target}: {error}") from error

    try:
        import soundfile  # type: ignore[import-not-found]
    except ImportError:
        soundfile = None  # type: ignore[assignment]

    if soundfile is not None:
        try:
            data, rate = soundfile.read(str(target))
            channels = 1 if getattr(data, "ndim", 1) == 1 else int(data.shape[1])
            duration = len(data) / float(rate) if rate else 0.0
            return AudioInfo(str(target), int(rate), channels, duration)
        except Exception as error:  # pragma: no cover - depends on file content
            raise AudioError(f"Cannot read audio file {target}: {error}") from error

    info = _probe_with_ffprobe(target)
    if info is not None:
        return info
    raise AudioError(f"Cannot probe {target}. Install soundfile or ffmpeg to read non-WAV audio.")


def _probe_with_ffprobe(target: Path) -> AudioInfo | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate,channels:format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=0",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - environment dependent
        return None

    values: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            try:
                values[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return AudioInfo(
        path=str(target),
        sample_rate=int(values.get("sample_rate", 0)),
        channels=int(values.get("channels", 0)),
        duration=float(values.get("duration", 0.0)),
    )


def is_normalized(path: str | Path) -> bool:
    """True when the file is already 16-bit PCM, 16 kHz, mono WAV."""
    target = Path(path)
    if target.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(target), "rb") as handle:
            return (
                handle.getframerate() == 16000
                and handle.getnchannels() == 1
                and handle.getsampwidth() == 2
            )
    except wave.Error:
        return False


def normalize_to_wav(
    source: str | Path,
    destination: str | Path,
    *,
    sample_rate: int = 16000,
    ffmpeg_bin: str = "ffmpeg",
) -> str:
    """Convert any supported audio file into 16 kHz mono 16-bit WAV.

    Returns the path of the normalized file. When the source already matches
    the target format the file is copied verbatim, so no external tool is
    required for that case.
    """
    src = Path(source)
    dst = Path(destination)
    if not src.exists():
        raise AudioError(f"Audio file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if is_normalized(src):
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return str(dst)

    if shutil.which(ffmpeg_bin) is None:
        raise AudioError(
            f"ffmpeg is required to convert {src.name} into 16 kHz mono WAV, "
            "but it was not found on PATH."
        )

    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:  # pragma: no cover - environment dependent
        raise AudioError(f"Failed to run ffmpeg: {error}") from error

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        message = detail[-1] if detail else f"exit code {completed.returncode}"
        raise AudioError(f"ffmpeg failed to convert {src.name}: {message}")
    return str(dst)


def read_wav_mono(path: str | Path) -> tuple[list[float], int]:
    """Read a PCM WAV file as mono float samples in [-1, 1]."""
    target = Path(path)
    try:
        with wave.open(str(target), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except wave.Error as error:
        raise AudioError(f"Cannot read WAV file {target}: {error}") from error

    if width != 2:
        raise AudioError(f"Only 16-bit PCM WAV is supported, got {width * 8}-bit.")

    import array

    samples = array.array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array.array(
            "h",
            [sum(samples[i : i + channels]) // channels for i in range(0, len(samples), channels)],
        )
    return [value / 32768.0 for value in samples], rate


def write_wav(path: str | Path, samples: list[float], sample_rate: int = 16000) -> str:
    """Write mono float samples to a 16-bit PCM WAV file."""
    import array

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clipped = array.array("h", [max(-32768, min(32767, int(value * 32768))) for value in samples])
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(clipped.tobytes())
    return str(target)


def slice_samples(samples: list[float], sample_rate: int, start: float, end: float) -> list[float]:
    """Extract the [start, end) second window from a mono sample list."""
    first = max(0, int(start * sample_rate))
    last = min(len(samples), int(end * sample_rate))
    return samples[first:last]
