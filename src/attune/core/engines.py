"""Transcription engines: one protocol, several interchangeable backends.

Every engine returns the same thing — a list of :class:`Utterance` whose
``engine_speaker`` is a **recording-local** key. Engines never decide who a
person is; that is the identity registry's job.

Heavy backends (FunASR, WhisperX) are imported lazily so that installing
``attune`` without the ASR extras stays cheap and the rest of the package
remains importable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models import Utterance


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Turns an audio file into timed, speaker-tagged utterances."""

    name: str

    def transcribe(self, audio_path: str) -> list[Utterance]:
        """Return the utterances found in ``audio_path``."""
        ...


def _as_embedding(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("embedding")
        if value is None:
            return None
    try:
        numbers = [float(item) for item in value]  # type: ignore[union-attr]
    except TypeError:
        return None
    if not numbers:
        return None
    return tuple(numbers)


class StaticEngine:
    """Returns a fixed utterance list. Used by tests and fixtures."""

    name = "static"

    def __init__(self, utterances: list[Utterance], name: str = "static") -> None:
        self._utterances = list(utterances)
        self.name = name

    def transcribe(self, audio_path: str) -> list[Utterance]:
        return list(self._utterances)


class JsonTranscriptEngine:
    """Loads utterances from a JSON file produced by another tool."""

    name = "json"

    def __init__(self, transcript_path: str | Path, name: str = "json") -> None:
        self._path = Path(transcript_path)
        self.name = name

    def transcribe(self, audio_path: str) -> list[Utterance]:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        rows = payload.get("utterances", payload) if isinstance(payload, dict) else payload
        utterances: list[Utterance] = []
        for row in rows:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            utterances.append(
                Utterance(
                    start=float(row.get("start", 0.0)),
                    end=float(row.get("end", 0.0)),
                    text=text,
                    engine_speaker=str(row.get("engine_speaker", "spk_0")),
                    embedding=_as_embedding(row.get("embedding")),
                )
            )
        return utterances


class FunASREngine:
    """Chinese-first ASR with speaker clustering, via FunASR.

    Uses ``paraformer-zh`` + ``fsmn-vad`` + ``ct-punc`` + ``cam++`` by default,
    which yields timed text, punctuation, and one voiceprint vector per
    speaker turn in a single pass.
    """

    def __init__(
        self,
        model: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        spk_model: str = "cam++",
        device: str = "cpu",
        max_speakers: int | None = None,
        model_obj: object | None = None,
        name: str = "funasr",
    ) -> None:
        self.model = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.device = device
        self.max_speakers = max_speakers
        self.name = name
        self._model_obj = model_obj

    def _build(self) -> object:
        try:
            from funasr import AutoModel  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "FunASR is not installed. Install it with: pip install 'attune[asr-funasr]'"
            ) from error

        kwargs: dict[str, object] = {"model": self.model, "device": self.device}
        if self.vad_model:
            kwargs["vad_model"] = self.vad_model
        if self.punc_model:
            kwargs["punc_model"] = self.punc_model
        if self.spk_model:
            kwargs["spk_model"] = self.spk_model
        if self.max_speakers:
            kwargs["max_speakers"] = self.max_speakers
        return AutoModel(**kwargs)

    def transcribe(self, audio_path: str) -> list[Utterance]:
        model = self._model_obj or self._build()
        generate = getattr(model, "generate", None)
        if generate is None:  # pragma: no cover - defensive
            raise RuntimeError("The FunASR model object does not expose generate().")

        result = generate(input=audio_path)
        payload = result[0] if isinstance(result, list) and result else result
        if not isinstance(payload, dict):
            return []

        sentences = payload.get("sentence_info") or []
        shared_embedding = _as_embedding(payload.get("spk_embedding"))
        utterances: list[Utterance] = []
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text", "")).strip()
            if not text:
                continue
            speaker = sentence.get("spk", index)
            embedding = _as_embedding(sentence.get("spk_embedding")) or shared_embedding
            utterances.append(
                Utterance(
                    start=float(sentence.get("start", 0.0)) / 1000.0,
                    end=float(sentence.get("end", 0.0)) / 1000.0,
                    text=text,
                    engine_speaker=f"spk_{speaker}",
                    embedding=embedding,
                )
            )
        return utterances


class WhisperXEngine:
    """Multilingual ASR with word-level alignment and pyannote diarization."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        hf_token: str | None = None,
        model_obj: object | None = None,
        name: str = "whisperx",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.hf_token = hf_token
        self.name = name
        self._model_obj = model_obj

    def _build(self) -> object:  # pragma: no cover - requires heavy deps
        try:
            import whisperx  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "WhisperX is not installed. Install it with: pip install 'attune[asr-whisperx]'"
            ) from error
        return whisperx.load_model(
            self.model_name, device=self.device, compute_type=self.compute_type
        )

    def transcribe(self, audio_path: str) -> list[Utterance]:  # pragma: no cover - heavy deps
        import whisperx  # type: ignore[import-not-found]

        model = self._model_obj or self._build()
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, language=self.language, batch_size=8)

        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language", self.language or "zh"),
            device=self.device,
        )
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, self.device)

        diarize_pipeline = whisperx.diarize.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )
        diarized = diarize_pipeline(audio)
        assigned = whisperx.assign_word_speakers(diarized, aligned)

        utterances: list[Utterance] = []
        for segment in assigned.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            speaker = segment.get("speaker", 0)
            utterances.append(
                Utterance(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=text,
                    engine_speaker=f"spk_{speaker}",
                )
            )
        return utterances
