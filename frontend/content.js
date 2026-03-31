// Injected into every page. Listens for messages from popup.

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "GET_VIDEO") {
    handleGetVideo(sendResponse);
    return true; // keep channel open for async
  }

  if (msg.action === "SHOW_BADGE") {
    showBadge(msg.verdict, msg.score);
    sendResponse({ ok: true });
  }

  if (msg.action === "SHOW_BADGE_ERROR") {
    showBadge("Error", null);
    sendResponse({ ok: true });
  }
});

// ── Find video and either return its src or record a blob clip ──────────────

function handleGetVideo(sendResponse) {
  const video = getBestVideo();

  if (!video) {
    sendResponse({ error: "No video element found on this page" });
    return;
  }

  const src = video.currentSrc || video.src;

  if (!src) {
    sendResponse({ error: "Video has no accessible source" });
    return;
  }

  // If it's a real URL (not blob), just return it
  if (!src.startsWith("blob:")) {
    sendResponse({ type: "url", url: src });
    return;
  }

  // It's a blob (YouTube, Twitter, etc) — record a short clip
  recordClip(video, sendResponse);
}

function getBestVideo() {
  const videos = Array.from(document.querySelectorAll("video"));
  if (!videos.length) return null;

  // Prefer the largest visible playing video
  const playing = videos.filter(v => !v.paused && !v.ended && v.readyState >= 2);
  const pool = playing.length ? playing : videos;

  return pool.reduce((best, v) => {
    const area = v.videoWidth * v.videoHeight;
    const bestArea = best.videoWidth * best.videoHeight;
    return area > bestArea ? v : best;
  });
}

function recordClip(video, sendResponse) {
  let stream;
  try {
    stream = video.captureStream();
  } catch (e) {
    sendResponse({ error: "Cannot capture stream from this video (cross-origin restricted)" });
    return;
  }

  const chunks = [];
  let recorder;

  try {
    recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8,opus" });
  } catch (e) {
    try {
      recorder = new MediaRecorder(stream);
    } catch (e2) {
      sendResponse({ error: "MediaRecorder not supported on this video" });
      return;
    }
  }

  recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const reader = new FileReader();
    reader.onloadend = () => {
      // Send base64 encoded clip back to popup
      sendResponse({ type: "blob", data: reader.result });
    };
    reader.readAsDataURL(blob);
  };

  recorder.onerror = () => {
    sendResponse({ error: "Recording failed" });
  };

  // Record 5 seconds
  recorder.start();
  setTimeout(() => {
    if (recorder.state !== "inactive") recorder.stop();
  }, 5000);
}

// ── Overlay badge ────────────────────────────────────────────────────────────

function showBadge(verdict, score) {
  // Remove existing badge
  const existing = document.getElementById("verifyy-badge");
  if (existing) existing.remove();

  const badge = document.createElement("div");
  badge.id = "verifyy-badge";

  const isFake = verdict === "Likely Fake";
  const isError = verdict === "Error";
  const percent = score != null ? (score * 100).toFixed(1) + "%" : "";

  const bg    = isError ? "#555"   : isFake ? "#c62828" : "#2e7d32";
  const label = isError ? "Verifyy: Error" : `Verifyy: ${verdict} ${percent}`;

  badge.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 2147483647;
    background: ${bg};
    color: #fff;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    cursor: pointer;
    transition: opacity 0.3s ease;
    letter-spacing: 0.3px;
  `;
  badge.innerText = label;

  // Click to dismiss
  badge.onclick = () => badge.remove();

  // Auto dismiss after 10s
  document.body.appendChild(badge);
  setTimeout(() => {
    if (badge.parentNode) {
      badge.style.opacity = "0";
      setTimeout(() => badge.remove(), 400);
    }
  }, 10000);
}