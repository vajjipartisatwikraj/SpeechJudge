"""Audio decoding and canonicalization (Requirement 5): mono, 16 kHz, 16-bit PCM range.

FFmpeg is used when available (required for MP3/OGG/WebM/M4A). Without FFmpeg a
pure-Python WAV fallback keeps development and tests working.
Temporary files live only inside a TemporaryDirectory that is always removed (Req 18.2).
"""
from __future__ import annotations

import base64
import binascii
import io
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.errors import AudioDecodeError, AudioLimitError
from app.core.logging import get_logger
from app.models.domain import CANONICAL_SAMPLE_RATE, AudioData

log = get_logger(__name__)

_DURATION_RE = re.compile(rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FFMPEG_TIMEOUT_S = 60


def sniff_format(raw: bytes) -> str | None:
    """Allow-list by magic bytes so FFmpeg never sees text playlists / concat scripts."""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    if raw[:3] == b"ID3" or (len(raw) > 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0):
        return "mp3/aac"
    if raw[:4] == b"OggS":
        return "ogg"
    if raw[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if raw[4:8] == b"ftyp":
        return "m4a"
    if raw[:4] == b"fLaC":
        return "flac"
    return None


def decode_base64_audio(value: str, max_bytes: int) -> bytes:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    value = value.strip()
    if not value:
        raise AudioDecodeError("Audio is empty")
    if len(value) * 3 // 4 > max_bytes + 4:
        raise AudioLimitError("audio limit exceeded")
    try:
        raw = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise AudioDecodeError("Audio is not valid base64") from exc
    if len(raw) > max_bytes:
        raise AudioLimitError("audio limit exceeded")
    if not raw:
        raise AudioDecodeError("Audio is empty")
    return raw


def _to_canonical(pcm16_mono: np.ndarray) -> np.ndarray:
    return (pcm16_mono.astype(np.float32) / 32768.0)


def _decode_ffmpeg(raw: bytes, settings: Settings) -> tuple[np.ndarray, float | None]:
    with tempfile.TemporaryDirectory(prefix="sj_") as tmp:
        src = Path(tmp) / "input.bin"
        src.write_bytes(raw)
        cmd = [
            settings.ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "info",
            "-protocol_whitelist", "file", "-i", str(src), "-vn",
            "-ac", "1", "-ar", str(CANONICAL_SAMPLE_RATE),
            "-t", str(settings.max_audio_seconds + 1.0),  # cap decode work
            "-f", "s16le", "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AudioDecodeError("Audio decoding timed out") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise AudioDecodeError("Audio is undecodable")
    pcm = np.frombuffer(proc.stdout[: len(proc.stdout) // 2 * 2], dtype="<i2")
    match = _DURATION_RE.search(proc.stderr)
    original = None
    if match:
        h, m, s = match.groups()
        original = int(h) * 3600 + int(m) * 60 + float(s)
    return pcm, original


def _decode_wav_fallback(raw: bytes) -> tuple[np.ndarray, float]:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            channels, width, rate, frames = (wf.getnchannels(), wf.getsampwidth(),
                                             wf.getframerate(), wf.getnframes())
            data = wf.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioDecodeError("Audio is undecodable") from exc
    if width == 2:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32)
    elif width == 1:
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
    elif width == 4:
        x = np.frombuffer(data, dtype="<i4").astype(np.float32) / 65536.0
    else:
        raise AudioDecodeError("Unsupported WAV sample width")
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    original = len(x) / float(rate) if rate else 0.0
    if rate != CANONICAL_SAMPLE_RATE and len(x) > 1:
        if rate > CANONICAL_SAMPLE_RATE:  # crude box low-pass to limit aliasing
            k = max(1, int(round(rate / CANONICAL_SAMPLE_RATE)))
            x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
        n_out = int(round(len(x) * CANONICAL_SAMPLE_RATE / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)
    return np.clip(np.round(x), -32768, 32767).astype(np.int16), original


def preprocess_audio(raw: bytes, settings: Settings) -> AudioData:
    """Decode arbitrary supported audio into canonical AudioData."""
    fmt = sniff_format(raw)
    if fmt is None:
        raise AudioDecodeError("Unsupported or unrecognized audio format")

    original: float | None
    if shutil.which(settings.ffmpeg_path):
        pcm, original = _decode_ffmpeg(raw, settings)
    elif fmt == "wav":
        pcm, original = _decode_wav_fallback(raw)
    else:
        raise AudioDecodeError(f"Cannot decode {fmt} audio: FFmpeg is not available")

    if len(pcm) == 0:
        raise AudioDecodeError("Audio contains no samples")
    canonical = len(pcm) / CANONICAL_SAMPLE_RATE
    if canonical > settings.max_audio_seconds:
        raise AudioLimitError("audio limit exceeded")
    return AudioData(samples=_to_canonical(pcm),
                     original_duration_s=original if original else canonical)
