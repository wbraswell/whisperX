import argparse
import gc
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, get_writer
from whisperx.log_utils import get_logger

logger = get_logger(__name__)

CHECKPOINT_FORMAT = "whisperx-stage-checkpoint-v1"
PARTIAL_TRANSCRIPTION_STAGE = "transcription-partial"


def build_stage_progress_callback(enabled, stage_number, total_stages, stage_name, minimum_increment=1.0):
    if not enabled:
        return None
    last_progress = [-1.0]

    def progress_callback(percent_complete):
        percent_complete = max(0.0, min(100.0, float(percent_complete)))
        if (
            last_progress[0] < 0.0
            or percent_complete >= 100.0 > last_progress[0]
            or percent_complete >= last_progress[0] + minimum_increment
        ):
            print(
                f"WhisperX stage {stage_number}/{total_stages}, {stage_name}: "
                f"{percent_complete:.2f}%",
                flush=True,
            )
            last_progress[0] = percent_complete

    return progress_callback


def checkpoint_path(checkpoint_dir, audio_path, stage):
    stem = Path(audio_path).stem
    return Path(checkpoint_dir) / f"{stem}.whisperx-{stage}.json"


def checkpoint_metadata(audio_path, stage, result):
    stat_result = os.stat(audio_path)
    return {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "audio_path": os.path.realpath(audio_path),
        "audio_size": stat_result.st_size,
        "audio_mtime_ns": stat_result.st_mtime_ns,
        "result": result,
    }


def write_checkpoint(checkpoint_dir, audio_path, stage, result):
    os.makedirs(checkpoint_dir, exist_ok=True)
    destination = checkpoint_path(checkpoint_dir, audio_path, stage)
    payload = checkpoint_metadata(audio_path, stage, result)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as checkpoint_file:
            json.dump(payload, checkpoint_file, ensure_ascii=False)
            checkpoint_file.write("\n")
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    logger.info(f"Saved {stage} checkpoint: {destination}")


def read_checkpoint(checkpoint_dir, audio_path, stage):
    source = checkpoint_path(checkpoint_dir, audio_path, stage)
    if not source.is_file():
        return None
    with source.open("r", encoding="utf-8") as checkpoint_file:
        payload = json.load(checkpoint_file)
    stat_result = os.stat(audio_path)
    expected = {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "audio_path": os.path.realpath(audio_path),
        "audio_size": stat_result.st_size,
        "audio_mtime_ns": stat_result.st_mtime_ns,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Checkpoint {source} is incompatible with the current input file "
                f"({key} mismatch). Remove it or use --resume_from none."
            )
    logger.info(f"Loaded {stage} checkpoint: {source}")
    return payload["result"]


def remove_checkpoint(checkpoint_dir, audio_path, stage):
    source = checkpoint_path(checkpoint_dir, audio_path, stage)
    try:
        source.unlink()
    except FileNotFoundError:
        return
    logger.info(f"Removed {stage} checkpoint: {source}")


def preflight_diarization_model(model_name, token, cache_dir):
    if not token:
        raise RuntimeError(
            "Diarization requires --hf_token or HF_TOKEN. No transcription has started."
        )
    logger.info(f"Preflighting gated diarization model access: {model_name}")
    try:
        hf_hub_download(
            repo_id=model_name,
            filename="config.yaml",
            token=token,
            cache_dir=cache_dir,
        )
    except Exception as error:
        raise RuntimeError(
            f"Cannot access diarization model {model_name}. Accept its Hugging Face "
            "user agreement and verify that the supplied token belongs to the same "
            "authorized account. No transcription has started."
        ) from error
    logger.info("Diarization model access preflight succeeded.")


def write_results(results, writer, writer_args, language):
    logger.info("Writing transcription output files...")
    for result, audio_path in results:
        result["language"] = result.get("language", language)
        writer(result, audio_path, writer_args)
    logger.info("Finished writing transcription output files.")


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    model_name = args.pop("model")
    batch_size = args.pop("batch_size")
    model_dir = args.pop("model_dir")
    model_cache_only = args.pop("model_cache_only")
    output_dir = args.pop("output_dir")
    output_format = args.pop("output_format")
    checkpoint_dir = args.pop("checkpoint_dir") or output_dir
    checkpoints_enabled = not args.pop("no_checkpoints")
    resume_from = args.pop("resume_from")
    stop_after = args.pop("stop_after")
    diarize_only = args.pop("diarize_only")
    skip_model_preflight = args.pop("skip_model_preflight")
    device = args.pop("device")
    device_index = args.pop("device_index")
    compute_type = args.pop("compute_type")
    verbose = args.pop("verbose")
    audio_paths = args.pop("audio")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    align_model_name = args.pop("align_model")
    interpolate_method = args.pop("interpolate_method")
    no_align = args.pop("no_align")
    task = args.pop("task")
    if task == "translate":
        no_align = True
    return_char_alignments = args.pop("return_char_alignments")

    hf_token = args.pop("hf_token")
    vad_method = args.pop("vad_method")
    vad_onset = args.pop("vad_onset")
    vad_offset = args.pop("vad_offset")
    chunk_size = args.pop("chunk_size")

    diarize = args.pop("diarize")
    min_speakers = args.pop("min_speakers")
    max_speakers = args.pop("max_speakers")
    diarize_model_name = args.pop("diarize_model")
    print_progress = args.pop("print_progress")
    return_speaker_embeddings = args.pop("speaker_embeddings")

    if diarize_only:
        diarize = True
        resume_from = "alignment"
    if resume_from in {"alignment", "diarization"} and no_align:
        parser.error("--resume_from alignment/diarization is incompatible with --no_align")
    if return_speaker_embeddings and not diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")
    if model_name.endswith(".en") and args["language"] != "en":
        args["language"] = "en"
    align_language = args["language"] or "en"

    temperature = args.pop("temperature")
    increment = args.pop("temperature_increment_on_fallback")
    temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment)) if increment is not None else [temperature]

    faster_whisper_threads = 4
    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "hotwords": args.pop("hotwords"),
        "suppress_tokens": [int(token) for token in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    writer_args = {argument: args.pop(argument) for argument in word_options}

    stage_names = ["voice activity detection", "transcription"]
    if not no_align:
        stage_names.append("alignment")
    if diarize:
        stage_names.append("speaker diarization")
    stage_numbers = {name: index + 1 for index, name in enumerate(stage_names)}
    total_stages = len(stage_names)
    vad_progress = build_stage_progress_callback(print_progress, stage_numbers["voice activity detection"], total_stages, "voice activity detection")
    transcription_progress = build_stage_progress_callback(print_progress, stage_numbers["transcription"], total_stages, "transcription", 0.0)
    alignment_progress = build_stage_progress_callback(print_progress, stage_numbers.get("alignment", 0), total_stages, "alignment") if not no_align else None
    diarization_progress = build_stage_progress_callback(print_progress, stage_numbers.get("speaker diarization", 0), total_stages, "speaker diarization") if diarize else None

    if diarize and not skip_model_preflight:
        preflight_diarization_model(diarize_model_name, hf_token, model_dir)

    if not no_align and not skip_model_preflight and resume_from in {"none", "auto", "transcription"}:
        logger.info("Preflighting alignment model before transcription...")
        preflight_align_model, preflight_align_metadata = load_align_model(
            align_language,
            device,
            model_name=align_model_name,
            model_dir=model_dir,
            model_cache_only=model_cache_only,
        )
        del preflight_align_model, preflight_align_metadata
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Alignment model preflight succeeded.")

    results = []
    remaining_audio_paths = []
    requested_resume_stage = resume_from
    if resume_from == "auto":
        requested_resume_stage = "diarization" if diarize else ("alignment" if not no_align else "transcription")

    for audio_path in audio_paths:
        resumed_result = None
        if checkpoints_enabled and requested_resume_stage != "none":
            candidate_stages = {
                "transcription": ["transcription"],
                "alignment": ["alignment", "transcription"],
                "diarization": ["diarization", "alignment", "transcription"],
            }[requested_resume_stage]
            for candidate_stage in candidate_stages:
                resumed_result = read_checkpoint(checkpoint_dir, audio_path, candidate_stage)
                if resumed_result is not None:
                    results.append((resumed_result, audio_path, candidate_stage))
                    break
        if resumed_result is None:
            remaining_audio_paths.append(audio_path)

    if remaining_audio_paths:
        logger.info("Loading transcription and voice activity detection models...")
        model = load_model(
            model_name,
            device=device,
            device_index=device_index,
            download_root=model_dir,
            compute_type=compute_type,
            language=args["language"],
            asr_options=asr_options,
            vad_method=vad_method,
            vad_options={"chunk_size": chunk_size, "vad_onset": vad_onset, "vad_offset": vad_offset},
            task=task,
            local_files_only=model_cache_only,
            threads=faster_whisper_threads,
            use_auth_token=hf_token,
        )
        logger.info("Finished loading transcription and voice activity detection models.")
        for audio_path in remaining_audio_paths:
            audio = load_audio(audio_path)
            partial_result = None
            if checkpoints_enabled:
                if requested_resume_stage == "none":
                    remove_checkpoint(
                        checkpoint_dir,
                        audio_path,
                        PARTIAL_TRANSCRIPTION_STAGE,
                    )
                else:
                    partial_result = read_checkpoint(
                        checkpoint_dir,
                        audio_path,
                        PARTIAL_TRANSCRIPTION_STAGE,
                    )
                    if partial_result is not None:
                        logger.info(
                            "Resuming transcription from "
                            f"{len(partial_result.get('segments', []))} completed chunks."
                        )

            def incremental_checkpoint(result):
                write_checkpoint(
                    checkpoint_dir,
                    audio_path,
                    PARTIAL_TRANSCRIPTION_STAGE,
                    result,
                )

            logger.info("Performing transcription...")
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                chunk_size=chunk_size,
                print_progress=False,
                verbose=verbose,
                progress_callback=transcription_progress,
                vad_progress_callback=vad_progress,
                initial_result=partial_result,
                checkpoint_callback=incremental_checkpoint if checkpoints_enabled else None,
            )
            if checkpoints_enabled:
                write_checkpoint(checkpoint_dir, audio_path, "transcription", result)
                remove_checkpoint(checkpoint_dir, audio_path, PARTIAL_TRANSCRIPTION_STAGE)
            results.append((result, audio_path, "transcription"))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if stop_after == "transcription":
        write_results([(result, path) for result, path, _stage in results], writer, writer_args, align_language)
        return

    if not no_align:
        needs_alignment = [(result, path, stage) for result, path, stage in results if stage == "transcription"]
        if needs_alignment:
            align_model, align_metadata = load_align_model(
                align_language,
                device,
                model_name=align_model_name,
                model_dir=model_dir,
                model_cache_only=model_cache_only,
            )
            updated_results = []
            for result, audio_path, stage in results:
                if stage != "transcription":
                    updated_results.append((result, audio_path, stage))
                    continue
                logger.info("Performing alignment...")
                if alignment_progress:
                    alignment_progress(0.0)
                aligned_result = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio_path,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=False,
                    progress_callback=alignment_progress,
                )
                aligned_result["language"] = result.get("language", align_language)
                if checkpoints_enabled:
                    write_checkpoint(checkpoint_dir, audio_path, "alignment", aligned_result)
                updated_results.append((aligned_result, audio_path, "alignment"))
            results = updated_results
            del align_model
            gc.collect()
            torch.cuda.empty_cache()

    if stop_after == "alignment":
        write_results([(result, path) for result, path, _stage in results], writer, writer_args, align_language)
        return

    diarization_error = None
    if diarize:
        logger.info("Performing diarization...")
        logger.info(f"Using model: {diarize_model_name}")
        try:
            diarize_model = DiarizationPipeline(
                model_name=diarize_model_name,
                token=hf_token,
                device=device,
                cache_dir=model_dir,
            )
            updated_results = []
            for result, audio_path, stage in results:
                if stage == "diarization":
                    updated_results.append((result, audio_path, stage))
                    continue
                diarize_result = diarize_model(
                    audio_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    return_embeddings=return_speaker_embeddings,
                    progress_callback=diarization_progress,
                )
                if return_speaker_embeddings:
                    diarize_segments, speaker_embeddings = diarize_result
                else:
                    diarize_segments = diarize_result
                    speaker_embeddings = None
                diarized_result = assign_word_speakers(diarize_segments, result, speaker_embeddings)
                if checkpoints_enabled:
                    write_checkpoint(checkpoint_dir, audio_path, "diarization", diarized_result)
                updated_results.append((diarized_result, audio_path, "diarization"))
            results = updated_results
        except Exception as error:
            diarization_error = error
            logger.exception(
                "Diarization failed. Completed transcription/alignment checkpoints and "
                "non-diarized outputs will still be preserved."
            )

    write_results([(result, path) for result, path, _stage in results], writer, writer_args, align_language)

    if diarization_error is not None:
        raise RuntimeError(
            "Diarization failed after earlier stages completed. The aligned outputs and "
            "checkpoints were preserved; rerun with --resume_from alignment after fixing "
            "the diarization problem."
        ) from diarization_error
