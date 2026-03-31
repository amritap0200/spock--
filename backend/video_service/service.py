import os
import cv2
import torch
import numpy as np
import subprocess
import time
import shutil
from pathlib import Path
from uuid import uuid4
from backend.celery_app import celery
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from backend.video_service.model import load_model
from backend.utils import DEVICE



model ,THRESHOLD= load_model()
model.to(DEVICE)
model.eval()

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

#frame extraction
def extract_frames(video_path, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "fps=1",
        "-frames:v", "5",
        str(work_dir / "frame_%03d.jpg")
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    if result.returncode != 0:
        return []

    frames= sorted([os.path.join(work_dir, f)
                     for f in os.listdir(work_dir)
                     if f.endswith(".jpg")
                     ])
    return frames

mtcnn = MTCNN(keep_all=False, device=DEVICE)
mtcnn_cpu = MTCNN(keep_all=False, device=torch.device("cpu"))

def detect_face(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        boxes, _ = mtcnn.detect(rgb_img)
    except RuntimeError as exc:
        if "Adaptive pool MPS" in str(exc):
            boxes, _ = mtcnn_cpu.detect(rgb_img)
        else:
            raise
    if boxes is None:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = boxes[0]

    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    face = image[y1:y2, x1:x2]
    if face is None or face.size == 0:
        return None

    return face

def preprocess(face_img):
    if face_img is None or face_img.size == 0:
        return None

    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0)

    return tensor.to(DEVICE)

def predict(tensor):
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
    return prob


def aggregate_scores(scores):
    return float(sum(scores) / len(scores))

def generate_gradcam(tensor, work_dir: Path):
    try:
        original_device = next(model.parameters()).device

        if DEVICE.type == "mps":
           model_cpu = model.cpu()
           target_layers = [model_cpu.target_layer]
           cam = GradCAM(model=model_cpu, target_layers=target_layers)
        else:
          target_layers = [model.target_layer]
          cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

        rgb_img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        heatmap_path = BASE_DIR / "media" / f"heatmap_{uuid4().hex}.jpg"
        cv2.imwrite(heatmap_path, heatmap)

        model.to(original_device)
        return heatmap_path

    except Exception as e:
        print("GradCAM failed:", e)
        return None
    

def analyze_video(video_path):
    start_time = time.time()
    resolved_video_path = Path(video_path)
    if not resolved_video_path.exists():
        candidate = BASE_DIR / video_path
        if candidate.exists():
            resolved_video_path = candidate
        else:
            return {
                "type": "video",
                "video_score": 0.0,
                "status": "Error",
                "error": f"File not found: {video_path}",
            }

    work_dir = TEMP_DIR / f"frames_{uuid4().hex}"
    frame_paths = extract_frames(str(resolved_video_path), work_dir)

    scores = []
    last_tensor = None

    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is None:
            continue

        face = detect_face(img)
        if face is None:
            continue

        tensor = preprocess(face)
        if tensor is None:
            continue
        score = predict(tensor)

        scores.append(score)
        last_tensor = tensor

    if len(scores) == 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {
            "type": "video",
            "video_score": 0.5,
            "status": "No Face Detected"
        }

    mean_score = float(np.mean(scores))
    trimmed_mean = mean_score
    if len(scores) >= 4:
        trimmed = sorted(scores)[1:-1]
        trimmed_mean = float(np.mean(trimmed))
    final_score = 0.85 * trimmed_mean + 0.15 * float(max(scores))
    margin = 0.05
    if final_score > THRESHOLD + margin:
        status = "Likely Fake"
    elif final_score < THRESHOLD - margin:
        status = "Likely Real"
    else:
        status = "Suspicious"

    result = {
        "type": "video",
        "video_score": final_score,
        "status": status,
        "frames_used": len(scores),
        "threshold": THRESHOLD,
    }

    if last_tensor is not None:
        heatmap_path = generate_gradcam(last_tensor, work_dir)
        if heatmap_path:
            result["heatmap"] = f"/media/{Path(heatmap_path).name}"
    else:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[VIDEO] Score: {final_score:.3f} | Time: {time.time() - start_time:.2f}s")

    return result



@celery.task(name="video_service.task_video_analysis")
def task_video_analysis(video_path):
    return analyze_video(video_path)