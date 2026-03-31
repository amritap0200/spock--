const API = "http://127.0.0.1:8000";

const statusEl        = document.getElementById("status");
const chooseBtn       = document.getElementById("chooseBtn");
const analyzeBtn      = document.getElementById("analyzeBtn");
const analyzePageBtn  = document.getElementById("analyzePageBtn");
const heatmapBtn      = document.getElementById("heatmapBtn");
const fileInput       = document.getElementById("videoInput");
const probabilityEl   = document.getElementById("probability");
const arrowEl         = document.getElementById("arrow");
const fakenessEl      = document.getElementById("fakenessValue");
const heatmapModal    = document.getElementById("heatmapModal");
const heatmapImage    = document.getElementById("heatmapImage");
const closeHeatmapBtn = document.getElementById("closeHeatmapBtn");

let latestHeatmapUrl = null;

// ── Gauge ────────────────────────────────────────────────────────────────────

function updateGauge(value) {
  const clamped = Math.max(0, Math.min(100, Number(value) || 0));
  const angle = -90 + (clamped / 100) * 180;
  arrowEl.style.transform = `rotate(${angle}deg)`;
  probabilityEl.innerText = `${Math.round(clamped)}%`;
}

function setResultValues(score01) {
  const score = Math.max(0, Math.min(1, Number(score01) || 0));
  updateGauge(0);
  requestAnimationFrame(() => updateGauge(score * 100));
  fakenessEl.innerText = `Fakeness: ${(score * 100).toFixed(1)}%`;
}

function setVerdict(verdict, finalScore) {
  const isFake = verdict === "Likely Fake";
  statusEl.innerText = `${verdict} — ${(finalScore * 100).toFixed(1)}%`;
  statusEl.style.color = isFake ? "#c62828" : "#2e7d32";
}

function setLoading(msg) {
  statusEl.innerText = msg;
  statusEl.style.color = "#333";
}

function setError(msg) {
  statusEl.innerText = msg;
  statusEl.style.color = "#c62828";
}

// ── Shared result handler ────────────────────────────────────────────────────

function handleResult(data, tabId, sendBadge) {
  setResultValues(data.final_score);
  setVerdict(data.verdict, data.final_score);
  console.log("Breakdown:", data.breakdown);

  if (data.heatmap_url) {
    latestHeatmapUrl = `${API}${data.heatmap_url}`;
    heatmapBtn.disabled = false;
  }

  if (sendBadge && tabId) {
    chrome.tabs.sendMessage(tabId, {
      action: "SHOW_BADGE",
      verdict: data.verdict,
      score: data.final_score,
    });
  }
}

// ── File upload flow ─────────────────────────────────────────────────────────

chooseBtn.onclick = () => fileInput.click();

fileInput.onchange = () => {
  const file = fileInput.files?.[0];
  statusEl.innerText = file ? `Selected: ${file.name}` : "No file selected";
  statusEl.style.color = "#333";
  latestHeatmapUrl = null;
  heatmapBtn.disabled = true;
  updateGauge(0);
  fakenessEl.innerText = "Fakeness: 0.0%";
};

analyzeBtn.onclick = async () => {
  const file = fileInput.files?.[0];
  if (!file) { setError("Please choose a file first"); return; }

  setLoading("Analyzing file...");
  analyzeBtn.disabled = true;
  heatmapBtn.disabled = true;
  latestHeatmapUrl = null;
  updateGauge(0);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API}/analyze`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);

    const data = await res.json();
    handleResult(data, null, false);
  } catch (err) {
    setError(`Error: ${err.message}`);
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
  }
};

// ── Analyze current page flow ────────────────────────────────────────────────

analyzePageBtn.onclick = async () => {
  setLoading("Finding video on page...");
  analyzePageBtn.disabled = true;
  heatmapBtn.disabled = true;
  latestHeatmapUrl = null;
  updateGauge(0);
  fakenessEl.innerText = "Fakeness: 0.0%";

  let tabId;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    tabId = tab.id;

    // Ask content script for the video
    const videoData = await new Promise((resolve, reject) => {
      chrome.tabs.sendMessage(tabId, { action: "GET_VIDEO" }, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(response);
        }
      });
    });

    if (videoData.error) {
      throw new Error(videoData.error);
    }

    let responseData;

    if (videoData.type === "url") {
      // Direct URL — send to /analyze-url
      setLoading("Sending video URL to backend...");
      const res = await fetch(`${API}/analyze-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoData.url }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
      responseData = await res.json();

    } else if (videoData.type === "blob") {
      // Blob/recorded clip — convert base64 back to file and send to /analyze
      setLoading("Uploading recorded clip...");
      const base64 = videoData.data;
      const byteStr = atob(base64.split(",")[1]);
      const ab = new ArrayBuffer(byteStr.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteStr.length; i++) ia[i] = byteStr.charCodeAt(i);
      const blob = new Blob([ab], { type: "video/webm" });
      const file = new File([blob], "page_clip.webm", { type: "video/webm" });

      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API}/analyze`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
      responseData = await res.json();
    }

    handleResult(responseData, tabId, true);

  } catch (err) {
    setError(`Error: ${err.message}`);
    console.error(err);
    if (tabId) {
      chrome.tabs.sendMessage(tabId, { action: "SHOW_BADGE_ERROR" });
    }
  } finally {
    analyzePageBtn.disabled = false;
  }
};

// ── Heatmap modal ────────────────────────────────────────────────────────────

heatmapBtn.onclick = () => {
  if (!latestHeatmapUrl) return;
  heatmapImage.src = latestHeatmapUrl;
  heatmapModal.classList.add("show");
};

closeHeatmapBtn.onclick = () => heatmapModal.classList.remove("show");
heatmapModal.onclick = (e) => {
  if (e.target === heatmapModal) heatmapModal.classList.remove("show");
};