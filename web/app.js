const LISTEN_MAX_MS = 12000;
const MIN_CLIP_S = 0.6;
const SEARCH_URL = "/search?decide=true&mode=windows&top_k=10";
const BAR_COUNT = 40;
const READY = new Set(["idle", "winner", "followup", "retry"]);

const state = {
  chunks: [],
  sampleRate: 48000,
  level: 0,
  heights: new Array(BAR_COUNT).fill(0),
  startedAt: 0,
  analyser: null,
  freq: null,
  context: null,
  stream: null,
  stopCapture: null,
  timer: 0,
  results: [],
  previewAudio: null,
  previewButton: null,
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function phase() {
  return document.body.dataset.phase;
}

function setCopy(headline, detail) {
  els.headline.textContent = headline;
  els.detail.textContent = detail || "";
}

function clearResults() {
  stopPreview();
  els.choices.replaceChildren();
}

function listenLabel(name) {
  if (name === "listening") return "Stop and search";
  if (name === "searching" || name === "arming") return "Searching";
  return "Tap to listen";
}

function setPhase(name) {
  document.body.dataset.phase = name;
  els.listen.disabled = name === "searching" || name === "arming";
  els.listen.setAttribute("aria-label", listenLabel(name));
}

function songLabel(item) {
  const title = (item.title || "").trim() || item.song_id || "this song";
  const artist = (item.artist || "").trim();
  return artist ? `${title} — ${artist}` : title;
}

function splitLabel(label) {
  const parts = String(label).split(" — ");
  if (parts.length < 2) return { title: label, artist: "" };
  return { title: parts[0], artist: parts.slice(1).join(" — ") };
}

function isUnsure(label) {
  const text = String(label).toLowerCase();
  return text.includes("not sure") || text.includes("none of these");
}

function youtubeUrl(title, artist) {
  const query = [title, artist].filter(Boolean).join(" ");
  return `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`;
}

function stopPreview() {
  if (state.previewAudio) {
    state.previewAudio.pause();
    state.previewAudio = null;
  }
  if (state.previewButton) {
    state.previewButton.textContent = "Preview";
    state.previewButton.classList.remove("is-playing");
    state.previewButton = null;
  }
}

function togglePreview(url, button) {
  if (state.previewButton === button && state.previewAudio && !state.previewAudio.paused) {
    stopPreview();
    return;
  }
  stopPreview();
  const audio = new Audio(url);
  state.previewAudio = audio;
  state.previewButton = button;
  button.textContent = "Pause";
  button.classList.add("is-playing");
  audio.addEventListener("ended", stopPreview);
  audio.play().catch(() => stopPreview());
}

function showIdle() {
  setPhase("idle");
  setCopy("Tap to Listen", "Hum a few seconds, then click again.");
  clearResults();
}

function showWinner(title, artist) {
  setPhase("winner");
  setCopy(title || "Match", artist || "");
  clearResults();
  if (title) appendSong({ title, artist }, null);
}

function showRetry(message) {
  setPhase("retry");
  setCopy("Hum again", message || "Try a clearer bit of the tune.");
  clearResults();
}

function appendUnsure(label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "unsure";
  button.textContent = label;
  button.addEventListener("click", () => choose(label));
  els.choices.appendChild(button);
}

function appendSong(item, confirmLabel) {
  const title = (item.title || "").trim() || "Match";
  const artist = (item.artist || "").trim();
  const row = document.createElement("article");
  row.className = "song";

  const slot = document.createElement("div");
  slot.className = "cover-slot";
  const cover = document.createElement("img");
  cover.className = "cover";
  cover.alt = "";
  cover.hidden = true;
  slot.appendChild(cover);

  const copy = document.createElement("div");
  copy.className = "song-copy";
  const titleEl = document.createElement(confirmLabel ? "button" : "h2");
  titleEl.className = "song-title";
  titleEl.textContent = title;
  if (confirmLabel) {
    titleEl.type = "button";
    titleEl.addEventListener("click", () => choose(confirmLabel));
  }
  const artistEl = document.createElement("p");
  artistEl.className = "song-artist";
  artistEl.textContent = artist;
  copy.append(titleEl, artistEl);

  const actions = document.createElement("div");
  actions.className = "actions";
  const preview = document.createElement("button");
  preview.type = "button";
  preview.className = "preview";
  preview.textContent = "Preview";
  preview.hidden = true;
  const youtube = document.createElement("a");
  youtube.className = "youtube";
  youtube.textContent = "YouTube";
  youtube.href = youtubeUrl(title, artist);
  youtube.target = "_blank";
  youtube.rel = "noopener noreferrer";
  actions.append(preview, youtube);

  row.append(slot, copy, actions);
  els.choices.appendChild(row);
  fillMeta(title, artist, cover, preview);
}

async function fillMeta(title, artist, cover, preview) {
  const url = `/meta?title=${encodeURIComponent(title)}&artist=${encodeURIComponent(artist)}`;
  try {
    const response = await fetch(url);
    if (!response.ok) return;
    const meta = await response.json();
    if (meta.cover) {
      cover.alt = "";
      cover.src = meta.cover;
      cover.hidden = false;
      cover.addEventListener("error", () => {
        cover.hidden = true;
        cover.removeAttribute("src");
      });
    }
    if (meta.preview) {
      preview.hidden = false;
      preview.addEventListener("click", () => togglePreview(meta.preview, preview));
    }
  } catch (_error) {
    preview.hidden = true;
  }
}

function showFollowup(decision) {
  const follow = decision.followup || {};
  const options = follow.options || [];
  setPhase("followup");
  setCopy("Which song?", follow.question || decision.message || "A few songs are close.");
  clearResults();
  options.forEach((label) => {
    if (isUnsure(label)) {
      appendUnsure(label);
      return;
    }
    const match = state.results.find((item) => songLabel(item) === label);
    const parts = match
      ? { title: (match.title || "").trim() || splitLabel(label).title, artist: (match.artist || "").trim() }
      : splitLabel(label);
    appendSong(parts, label);
  });
}

function choose(label) {
  if (isUnsure(label)) {
    showRetry("Hum a different part of the tune.");
    return;
  }
  const match = state.results.find((item) => songLabel(item) === label);
  if (match) {
    showWinner((match.title || "").trim() || label, (match.artist || "").trim());
    return;
  }
  const parts = splitLabel(label);
  showWinner(parts.title, parts.artist);
}

function applyDecision(payload) {
  const decision = payload.decision || {};
  state.results = payload.results || decision.results || [];
  if (decision.action === "show") {
    const top = state.results[0] || {};
    showWinner((top.title || "").trim() || decision.message || "Match", (top.artist || "").trim());
    return;
  }
  if (decision.action === "ask_followup") {
    showFollowup(decision);
    return;
  }
  showRetry(decision.message || "Hum again, a little clearer.");
}

function formData(blob) {
  const data = new FormData();
  data.append("audio", blob, "hum.wav");
  return data;
}

async function searchClip(blob) {
  setPhase("searching");
  setCopy("Searching", "Matching the melody.");
  clearResults();
  try {
    const response = await fetch(SEARCH_URL, { method: "POST", body: formData(blob) });
    if (!response.ok) throw new Error(String(response.status));
    applyDecision(await response.json());
  } catch (_error) {
    showRetry("The search did not answer. Check that it is running, then hum again.");
  }
}

function rms(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function keepSamples(samples) {
  const copy = samples instanceof Float32Array ? samples : new Float32Array(samples);
  state.chunks.push(copy);
  state.level = rms(copy);
}

async function openMic() {
  const bare = { audio: true };
  const clean = {
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  };
  try {
    return await navigator.mediaDevices.getUserMedia(clean);
  } catch (error) {
    if (error.name === "OverconstrainedError" || error.name === "NotFoundError") {
      return navigator.mediaDevices.getUserMedia(bare);
    }
    throw error;
  }
}

function attachAnalyser(context, source) {
  const analyser = context.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.7;
  source.connect(analyser);
  state.analyser = analyser;
  state.freq = new Uint8Array(analyser.frequencyBinCount);
}

async function attachWorklet(context, source) {
  await context.audioWorklet.addModule("/capture-worklet.js");
  const node = new AudioWorkletNode(context, "hum-capture");
  node.port.onmessage = (event) => keepSamples(event.data);
  const mute = context.createGain();
  mute.gain.value = 0;
  source.connect(node);
  node.connect(mute);
  mute.connect(context.destination);
  return () => {
    node.disconnect();
    mute.disconnect();
  };
}

function attachScriptProcessor(context, source) {
  const processor = context.createScriptProcessor(4096, 1, 1);
  const mute = context.createGain();
  mute.gain.value = 0;
  processor.onaudioprocess = (event) => {
    keepSamples(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(mute);
  mute.connect(context.destination);
  return () => {
    processor.disconnect();
    mute.disconnect();
  };
}

async function attachCapture(context, source) {
  try {
    return await attachWorklet(context, source);
  } catch (_error) {
    return attachScriptProcessor(context, source);
  }
}

function mergeSamples(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function stopTracks() {
  if (state.timer) window.clearTimeout(state.timer);
  state.timer = 0;
  if (state.stopCapture) state.stopCapture();
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  if (state.context) state.context.close();
  state.stopCapture = null;
  state.stream = null;
  state.context = null;
  state.analyser = null;
  state.level = 0;
}

async function beginListening() {
  if (!READY.has(phase())) return;
  setPhase("arming");
  clearResults();
  setCopy("Listening", "");
  try {
    const stream = await openMic();
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    state.chunks = [];
    state.sampleRate = context.sampleRate;
    state.stream = stream;
    state.context = context;
    state.startedAt = performance.now();
    attachAnalyser(context, source);
    state.stopCapture = await attachCapture(context, source);
    setPhase("listening");
    setCopy("Listening", "0s · click again to search");
    state.timer = window.setTimeout(() => finishListening(), LISTEN_MAX_MS);
  } catch (_error) {
    stopTracks();
    showRetry("The microphone is blocked. Allow it in the browser, then try again.");
  }
}

async function finishListening() {
  if (phase() !== "listening") return;
  setPhase("arming");
  const seconds = (performance.now() - state.startedAt) / 1000;
  const samples = mergeSamples(state.chunks);
  const rate = state.sampleRate;
  stopTracks();
  if (seconds < MIN_CLIP_S || samples.length < rate * MIN_CLIP_S) {
    showRetry("Hum a little longer.");
    return;
  }
  await searchClip(encodeWav(samples, rate));
}

function onListenClick() {
  if (phase() === "listening") {
    finishListening();
    return;
  }
  if (READY.has(phase())) beginListening();
}

function tickClock() {
  if (phase() !== "listening") return;
  const seconds = Math.max(0, Math.round((performance.now() - state.startedAt) / 1000));
  els.detail.textContent = `${seconds}s · click again to search`;
}

function barTargets() {
  const quiet = new Array(BAR_COUNT).fill(0);
  if (phase() !== "listening" || !state.analyser || !state.freq) return quiet;
  state.analyser.getByteFrequencyData(state.freq);
  const level = Math.min(1, state.level * 7);
  const usable = Math.max(1, Math.floor(state.freq.length * 0.7));
  const targets = [];
  const now = performance.now();
  for (let i = 0; i < BAR_COUNT; i += 1) {
    const index = Math.min(usable - 1, Math.floor((i / BAR_COUNT) * usable));
    const shape = state.freq[index] / 255;
    const wave = reducedMotion() ? 0.5 : 0.5 + 0.5 * Math.sin(now / 240 + i * 0.55);
    const breathe = 0.34 + wave * 0.28;
    const audio = level * (0.3 + 0.7 * shape);
    targets.push(Math.min(1, breathe + audio * 0.85));
  }
  return targets;
}

function ringGeometry() {
  const canvas = els.bars;
  const dpr = canvas.clientWidth ? canvas.width / canvas.clientWidth : 1;
  const orb = (els.listen.offsetWidth / 2) * dpr;
  const limit = Math.min(canvas.width, canvas.height) / 2 - 4 * dpr;
  return {
    dpr,
    inner: orb + 10 * dpr,
    maxLen: Math.max(12 * dpr, limit - orb - 10 * dpr),
  };
}

function drawBars() {
  const canvas = els.bars;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const { dpr, inner, maxLen } = ringGeometry();
  const cx = width / 2;
  const cy = height / 2;
  ctx.lineCap = "round";
  ctx.lineWidth = Math.max(2.5, 3.5 * dpr);
  for (let i = 0; i < BAR_COUNT; i += 1) {
    const amount = state.heights[i];
    if (amount < 0.04) continue;
    const angle = -Math.PI / 2 + (i / BAR_COUNT) * Math.PI * 2;
    const length = 8 * dpr + amount * maxLen;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    ctx.strokeStyle = `rgba(0, 0, 0, ${0.45 + 0.55 * amount})`;
    ctx.beginPath();
    ctx.moveTo(cx + cos * inner, cy + sin * inner);
    ctx.lineTo(cx + cos * (inner + length), cy + sin * (inner + length));
    ctx.stroke();
  }
}

function resizeCanvas() {
  const canvas = els.bars;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
}

function tick() {
  const targets = barTargets();
  const ease = reducedMotion() ? 0.85 : 0.28;
  let energy = 0;
  for (let i = 0; i < BAR_COUNT; i += 1) {
    state.heights[i] += (targets[i] - state.heights[i]) * ease;
    energy += state.heights[i];
  }
  els.stage.style.setProperty("--level", (energy / BAR_COUNT).toFixed(3));
  drawBars();
  tickClock();
  window.requestAnimationFrame(tick);
}

function boot() {
  els.headline = $("headline");
  els.detail = $("detail");
  els.listen = $("listen");
  els.bars = $("bars");
  els.choices = $("choices");
  els.stage = $("stage");
  els.listen.addEventListener("click", onListenClick);
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  window.requestAnimationFrame(tick);
}

boot();
