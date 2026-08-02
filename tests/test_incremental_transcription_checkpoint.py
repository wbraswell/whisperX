"""Tests for resumable per-chunk transcription checkpoints."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from whisperx.asr import FasterWhisperPipeline, Pyannote


VAD_SEGMENTS = [
    {"start": 0.0, "end": 1.0},
    {"start": 1.0, "end": 2.0},
]


class FakeVadModel:
    def __call__(self, _audio, **_kwargs):
        return []


class FakePipeline:
    transcribe = FasterWhisperPipeline.transcribe

    def __init__(self, outputs):
        self.vad_model = FakeVadModel()
        self._vad_params = {"vad_onset": 0.5, "vad_offset": 0.363}
        self.tokenizer = SimpleNamespace(language_code="en", task="transcribe")
        self.preset_language = "en"
        self.suppress_numerals = False
        self._batch_size = 1
        self.outputs = list(outputs)
        self.input_count = 0

    def __call__(self, inputs, batch_size, num_workers):
        assert batch_size == 1
        assert num_workers == 0
        for input_data, output in zip(inputs, self.outputs):
            assert "inputs" in input_data
            self.input_count += 1
            yield {
                "text": [output],
                "avg_logprob": [-0.1],
            }


def configure_vad(monkeypatch):
    monkeypatch.setattr(Pyannote, "preprocess_audio", staticmethod(lambda audio: audio))
    monkeypatch.setattr(
        Pyannote,
        "merge_chunks",
        staticmethod(lambda _segments, _chunk_size, onset, offset: deepcopy(VAD_SEGMENTS)),
    )


def test_checkpoint_callback_receives_each_completed_chunk(monkeypatch):
    configure_vad(monkeypatch)
    pipeline = FakePipeline(["first", "second"])
    checkpoints = []
    progress = []

    result = pipeline.transcribe(
        np.zeros(2 * 16000, dtype=np.float32),
        language="en",
        checkpoint_callback=lambda checkpoint: checkpoints.append(deepcopy(checkpoint)),
        progress_callback=progress.append,
    )

    assert pipeline.input_count == 2
    assert [len(checkpoint["segments"]) for checkpoint in checkpoints] == [1, 2]
    assert checkpoints[-1] == result
    assert progress == [0.0, 50.0, 100.0]


def test_resume_skips_already_checkpointed_chunks(monkeypatch):
    configure_vad(monkeypatch)
    pipeline = FakePipeline(["second"])
    checkpoints = []
    progress = []
    initial_result = {
        "segments": [
            {
                "text": "first",
                "start": 0.0,
                "end": 1.0,
                "avg_logprob": -0.1,
            }
        ],
        "language": "en",
    }

    result = pipeline.transcribe(
        np.zeros(2 * 16000, dtype=np.float32),
        language="en",
        initial_result=initial_result,
        checkpoint_callback=lambda checkpoint: checkpoints.append(deepcopy(checkpoint)),
        progress_callback=progress.append,
    )

    assert pipeline.input_count == 1
    assert [segment["text"] for segment in result["segments"]] == ["first", "second"]
    assert len(checkpoints) == 1
    assert checkpoints[0] == result
    assert progress == [50.0, 100.0]


def test_resume_rejects_changed_vad_boundaries(monkeypatch):
    configure_vad(monkeypatch)
    pipeline = FakePipeline(["second"])
    initial_result = {
        "segments": [
            {
                "text": "first",
                "start": 0.0,
                "end": 0.5,
                "avg_logprob": -0.1,
            }
        ],
        "language": "en",
    }

    with pytest.raises(RuntimeError, match="does not match"):
        pipeline.transcribe(
            np.zeros(2 * 16000, dtype=np.float32),
            language="en",
            initial_result=initial_result,
        )


def test_resume_rejects_changed_language(monkeypatch):
    configure_vad(monkeypatch)
    pipeline = FakePipeline(["second"])
    initial_result = {
        "segments": [
            {
                "text": "first",
                "start": 0.0,
                "end": 1.0,
                "avg_logprob": -0.1,
            }
        ],
        "language": "es",
    }

    with pytest.raises(RuntimeError, match="language does not match"):
        pipeline.transcribe(
            np.zeros(2 * 16000, dtype=np.float32),
            language="en",
            initial_result=initial_result,
        )
