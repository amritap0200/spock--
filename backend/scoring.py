def compute_final_score(video_result, audio_result, metadata_result):
    v = video_result.get("video_score", 0.5)
    a = audio_result.get("audio_probability", 0.5)
    m = metadata_result.get("metadata_score", 0.0)

    v = max(0.0, min(1.0, v))
    a = max(0.0, min(1.0, a))
    m = max(0.0, min(1.0, m))

    final_score = 0.6 * v + 0.4 * a

    if metadata_result.get("recycled"):
        final_score = max(final_score, 0.9)

    final_score = max(0.0, min(1.0, final_score))

    verdict = "Likely Fake" if final_score > 0.50 else "Likely Real"

    return {
        "final_score": round(final_score, 4),
        "verdict": verdict,
        "breakdown": {
            "video": round(v, 4),
            "audio": round(a, 4),
            "metadata": round(m, 4),
        }
    }