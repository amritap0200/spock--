# backend/main.py
import asyncio
from pathlib import Path
from uuid import uuid4
import httpx
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult

from backend.celery_app import celery
from backend.video_service import task_video_analysis
from backend.audio_service import task_audio_analysis
from backend.metadata_service import task_metadata_analysis
from backend.scoring import compute_final_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
MEDIA_DIR = BASE_DIR / "media"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")


@app.get("/")
def root():
    return {"status": "Verifyy backend running"}


def _enqueue_analysis(video_path: str, analysis_id: str | None = None):
    analysis_id = analysis_id or str(uuid4())
    video_task = task_video_analysis.delay(video_path)
    audio_task = task_audio_analysis.delay(video_path)
    metadata_task = task_metadata_analysis.delay(video_path)
    return {
        "analysis_id": analysis_id,
        "video_task_id": video_task.id,
        "audio_task_id": audio_task.id,
        "metadata_task_id": metadata_task.id,
    }

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    analysis_id: str | None = Form(default=None),
):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    upload_dir = TEMP_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"upload_{uuid4().hex}{suffix}"

    try:
        content = await file.read()
        temp_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to save upload: {exc}")

    queued = _enqueue_analysis(str(temp_path.resolve()), analysis_id=analysis_id)

    try:
        video_result    = AsyncResult(queued["video_task_id"],    app=celery).get(timeout=180)
        audio_result    = AsyncResult(queued["audio_task_id"],    app=celery).get(timeout=180)
        metadata_result = AsyncResult(queued["metadata_task_id"], app=celery).get(timeout=180)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis task failed: {exc}")

    final = compute_final_score(video_result, audio_result, metadata_result)

    heatmap_url = None
    heatmap_path = video_result.get("heatmap")
    if heatmap_path:
        heatmap_url = f"/media/{Path(heatmap_path).name}"

    return {
        "analysis_id":  queued["analysis_id"],
        "final_score":  final["final_score"],
        "verdict":      final["verdict"],
        "breakdown":    final["breakdown"],
        "heatmap_url":  heatmap_url,
    }

@app.post("/analyze-url")
async def analyze_url(request: Request):
    body = await request.json()
    video_url = body.get("url")
    if not video_url:
        raise HTTPException(status_code=400, detail="No URL provided")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(video_url, follow_redirects=True)
            r.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch video: {exc}")

    suffix = ".mp4"
    upload_dir = TEMP_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"upload_{uuid4().hex}{suffix}"
    temp_path.write_bytes(r.content)

    queued = _enqueue_analysis(str(temp_path.resolve()))

    try:
        video_result    = AsyncResult(queued["video_task_id"],    app=celery).get(timeout=180)
        audio_result    = AsyncResult(queued["audio_task_id"],    app=celery).get(timeout=180)
        metadata_result = AsyncResult(queued["metadata_task_id"], app=celery).get(timeout=180)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    final = compute_final_score(video_result, audio_result, metadata_result)
    heatmap_url = None
    heatmap_path = video_result.get("heatmap")
    if heatmap_path:
        heatmap_url = f"/media/{Path(heatmap_path).name}"

    return {
        "analysis_id": queued["analysis_id"],
        "final_score": final["final_score"],
        "verdict":     final["verdict"],
        "breakdown":   final["breakdown"],
        "heatmap_url": heatmap_url,
    }


@app.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        video_id    = data["video_task_id"]
        audio_id    = data["audio_task_id"]
        metadata_id = data["metadata_task_id"]

        video_result    = AsyncResult(video_id,    app=celery)
        audio_result    = AsyncResult(audio_id,    app=celery)
        metadata_result = AsyncResult(metadata_id, app=celery)

        sent_video = sent_audio = sent_meta = False
        start_time = asyncio.get_event_loop().time()

        while True:
            if asyncio.get_event_loop().time() - start_time > 300:
                await websocket.send_json({"stage": "error", "result": {"message": "Timed out"}})
                break

            if video_result.failed() or audio_result.failed() or metadata_result.failed():
                await websocket.send_json({"stage": "error", "result": {"message": "Task failed"}})
                break

            if video_result.ready() and not sent_video:
                await websocket.send_json({"stage": "video_complete", "result": video_result.result})
                sent_video = True

            if audio_result.ready() and not sent_audio:
                await websocket.send_json({"stage": "audio_complete", "result": audio_result.result})
                sent_audio = True

            if metadata_result.ready() and not sent_meta:
                await websocket.send_json({"stage": "metadata_complete", "result": metadata_result.result})
                sent_meta = True

            if sent_video and sent_audio and sent_meta:
                final = compute_final_score(
                    video_result.result,
                    audio_result.result,
                    metadata_result.result,
                )
                heatmap_url = None
                heatmap_path = video_result.result.get("heatmap")
                if heatmap_path:
                    heatmap_url = f"/media/{Path(heatmap_path).name}"

                await websocket.send_json({
                    "stage": "final",
                    "result": {**final, "heatmap_url": heatmap_url}
                })
                break

            await asyncio.sleep(0.5)

    except Exception:
        pass
    finally:
        await websocket.close()