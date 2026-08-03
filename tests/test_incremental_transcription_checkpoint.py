"""Tests for resumable per-chunk transcription checkpoints."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from whisperx.asr import FasterWhisperPipeline, Pyannote

TEST_VERSION = "0.003"


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


def test_stage_checkpoint_survives_path_and_mtime_changes(tmp_path):
    from whisperx.transcribe import read_checkpoint, write_checkpoint

    original_directory = tmp_path / "original"
    restored_directory = tmp_path / "restored"
    checkpoint_directory = tmp_path / "checkpoints"
    original_directory.mkdir()
    restored_directory.mkdir()
    original_audio = original_directory / "meeting.mp4"
    restored_audio = restored_directory / "meeting.mp4"
    original_audio.write_bytes(b"portable-checkpoint-audio")
    restored_audio.write_bytes(original_audio.read_bytes())
    restored_audio.touch()

    expected_result = {"segments": [{"text": "hello"}], "language": "en"}
    write_checkpoint(checkpoint_directory, original_audio, "transcription", expected_result)

    assert read_checkpoint(
        checkpoint_directory,
        restored_audio,
        "transcription",
    ) == expected_result


def test_stage_checkpoint_rejects_same_size_different_audio(tmp_path):
    from whisperx.transcribe import read_checkpoint, write_checkpoint

    original_directory = tmp_path / "original"
    restored_directory = tmp_path / "restored"
    checkpoint_directory = tmp_path / "checkpoints"
    original_directory.mkdir()
    restored_directory.mkdir()
    original_audio = original_directory / "meeting.mp4"
    restored_audio = restored_directory / "meeting.mp4"
    original_audio.write_bytes(b"aaaaaaaa")
    restored_audio.write_bytes(b"bbbbbbbb")

    write_checkpoint(
        checkpoint_directory,
        original_audio,
        "transcription",
        {"segments": [], "language": "en"},
    )

    with pytest.raises(RuntimeError, match="audio_sha256 mismatch"):
        read_checkpoint(checkpoint_directory, restored_audio, "transcription")


def test_checkpoint_hook_runs_after_atomic_checkpoint_write(tmp_path, monkeypatch):
    from whisperx.transcribe import write_checkpoint

    audio_path = tmp_path / "meeting.mp4"
    checkpoint_directory = tmp_path / "checkpoints"
    hook_output = tmp_path / "hook-output.txt"
    hook_script = tmp_path / "checkpoint-hook.sh"
    audio_path.write_bytes(b"hook-test-audio")
    hook_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s' \"$1\" > {hook_output}\n",
        encoding="utf-8",
    )
    hook_script.chmod(0o700)
    monkeypatch.setenv("WHISPERX_CHECKPOINT_HOOK", str(hook_script))

    write_checkpoint(
        checkpoint_directory,
        audio_path,
        "transcription",
        {"segments": [], "language": "en"},
    )

    assert hook_output.read_text(encoding="utf-8") == str(
        checkpoint_directory / "meeting.whisperx-transcription.json"
    )
