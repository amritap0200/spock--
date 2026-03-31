from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.video_service import analyze_video
from backend.audio_service import analyze_audio
from backend.metadata_service import analyze_metadata
from backend.scoring import compute_final_score

video_path = str(Path(__file__).resolve().parent / "media" / "test_with_audio.mp4")

print("\nRunning full pipeline locally...\n")

video_result = analyze_video(video_path)
print("VIDEO RESULT:", video_result)

audio_result = analyze_audio(video_path)
print("AUDIO RESULT:", audio_result)

metadata_result = analyze_metadata(video_path)
print("METADATA RESULT:", metadata_result)

final = compute_final_score(video_result, audio_result, metadata_result)
print("\nFINAL FUSION RESULT:", final)
