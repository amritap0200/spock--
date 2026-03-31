import os
import subprocess

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import librosa
import numpy as np
import torch
import time
from pathlib import Path
from backend.celery_app import celery
from backend.audio_service.model import load_audio_model

DEVICE = torch.device("cpu")
torch.set_num_threads(2)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

model = None
THRESHOLD = None
MODEL_LOAD_ERROR = None
try:
    model, THRESHOLD = load_audio_model()
    model.to(DEVICE)
    model.eval()
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)


# --------------------
# Extract audio from video
# --------------------
def extract_audio(video_path, output_path):
    # First check if the video has an audio stream
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if "audio" not in probe_result.stdout:
        raise ValueError("No audio stream found in video")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-t", "10",
        "-vn",
        output_path
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return output_path

def load_audio(path):
    audio, sr = librosa.load(path, sr=16000)
    return audio, sr


def create_mel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # CRITICAL
    mel_db = np.nan_to_num(mel_db)

    return mel_db

def preprocess(mel_db):
    mel_tensor = torch.tensor(mel_db).float()
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,128,T)

    return mel_tensor.to(DEVICE)


def analyze_audio(video_path):
    if model is None:
        return {
            "type": "audio",
            "audio_probability": 0.0,
            "status": "Error",
            "error": f"Audio model unavailable: {MODEL_LOAD_ERROR}",
        }

    start = time.time()
    resolved_video_path = Path(video_path)
    if not resolved_video_path.exists():
        candidate = BASE_DIR / video_path
        if candidate.exists():
            resolved_video_path = candidate
        else:
            return {
                "type": "audio",
                "audio_probability": 0.0,
                "status": "Error",
                "error": f"File not found: {video_path}",
            }

    wav_path = str(TEMP_DIR / "temp.wav")

    try:
        extract_audio(str(resolved_video_path), wav_path)
    except ValueError as ve:
        # No audio stream — not an error, just return neutral score
        return {
            "type": "audio",
            "audio_probability": 0.5,
            "status": "No Audio",
            "error": str(ve),
        }
    except subprocess.CalledProcessError:
        return {
            "type": "audio",
            "audio_probability": 0.0,
            "status": "Error",
            "error": "ffmpeg failed to extract audio",
        }
    
    try:
        audio, sr = load_audio(wav_path)

        if sr != 16000:
            return {
                "type": "audio",
                "audio_probability": 0.0,
                "status": "Error",
                "error": "Sample rate mismatch",
            }

        mel = create_mel(audio, sr)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        MAX_LEN = 300

        if mel.shape[1] < MAX_LEN:
            pad_size = MAX_LEN - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad_size)), mode='constant')
        else:
            mel = mel[:, :MAX_LEN]

        tensor = preprocess(mel)

        with torch.no_grad():
            prob = model(tensor).item()
    except Exception as exc:
        return {
            "type": "audio",
            "audio_probability": 0.0,
            "status": "Error",
            "error": f"Audio analysis failed: {exc}",
        }

    if prob > THRESHOLD + 0.02:
      status = "Likely Fake"
    elif prob < THRESHOLD - 0.02:
      status = "Likely Real"
    else:
      status = "Uncertain"
    
    print(f"[AUDIO] Score: {prob:.3f} | Time: {time.time() - start:.2f}s")

    return {
        "type": "audio",
        "audio_probability": prob,
        "status":status
    }


@celery.task(name="audio_service.task_audio_analysis")
def task_audio_analysis(video_path):
    return analyze_audio(video_path)
