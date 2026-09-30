const LISTEN_MAX_MS = 12000;
const MIN_CLIP_S = 0.6;
const SEARCH_URL = "/search?decide=true&mode=windows&top_k=10";
const BAR_COUNT = 16;
const BAR_COLORS = ["#ff375f", "#bf5af2", "#5e5ce6", "#64d2ff", "#30d158"];

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
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function setCopy(headline, detail) {
  els.headline.textContent = headline;
  els.detail.textContent = detail || "";
}

function clearChoices() {
  els.choices.replaceChildren();
}

function buttonLabel(phase) {
  if (phase === "listening") return "Stop";
  if (phase === "searching") return "Searching";
  return "Listen";
}

function setPhase(phase) {
  document.body.dataset.phase = phase;
  els.listen.disabled = phase === "searching" || phase === "arming";
  els.listen.textContent = buttonLabel(phase);
  els.listen.setAttribute("aria-label", buttonLabel(phase));
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

function showIdle() {
  setPhase("idle");
  setCopy("Hum a few seconds.", "Tap to listen.");
  clearChoices();
}

function showWinner(title, artist) {
  setPhase("idle");
  setCopy(title || "Match", artist || "");
  clearChoices();
}

function showRetry(message) {
  setPhase("idle");
  setCopy("Hum again.", message || "Try a clearer bit of the tune.");
  clearChoices();
}

function showFollowup(decision) {
  const follow = decision.followup || {};
  const options = follow.options || [];
  setPhase("idle");
  setCopy("Which song?", follow.question || decision.message || "A few songs are close.");
  clearChoices();
  options.forEach((label, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pressable";
    button.textContent = label;
    button.style.animationDelay = `${index * 40}ms`;
    button.addEventListener("click", () => choose(label));
    els.choices.appendChild(button);
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
  setCopy("Searching.", "Matching that melody.");
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
  if (document.body.dataset.phase !== "idle") return;
  setPhase("arming");
  clearChoices();
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
    setCopy("Listening.", "0s. Tap again to search.");
    state.timer = window.setTimeout(() => finishListening(), LISTEN_MAX_MS);
  } catch (_error) {
    stopTracks();
    showRetry("The microphone is blocked. Allow it in the browser, then try again.");
  }
}

async function finishListening() {
  if (document.body.dataset.phase !== "listening") return;
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
  if (document.body.dataset.phase === "listening") {
    finishListening();
    return;
  }
  if (document.body.dataset.phase === "idle") beginListening();
}

function tickClock() {
  if (document.body.dataset.phase !== "listening") return;
  const seconds = Math.max(0, Math.round((performance.now() - state.startedAt) / 1000));
  els.detail.textContent = `${seconds}s. Tap again to search.`;
}

function barTargets() {
  const quiet = new Array(BAR_COUNT).fill(0);
  if (document.body.dataset.phase !== "listening" || !state.analyser || !state.freq) return quiet;
  state.analyser.getByteFrequencyData(state.freq);
  const level = Math.min(1, state.level * 7);
  const usable = Math.max(1, Math.floor(state.freq.length * 0.7));
  const targets = [];
  for (let i = 0; i < BAR_COUNT; i += 1) {
    const index = Math.min(usable - 1, Math.floor((i / BAR_COUNT) * usable));
    const shape = state.freq[index] / 255;
    targets.push(0.06 + level * (0.35 + 0.65 * shape) * 0.94);
  }
  return targets;
}

function mixColor(t) {
  const scaled = Math.min(0.999, Math.max(0, t)) * (BAR_COLORS.length - 1);
  const index = Math.floor(scaled);
  const frac = scaled - index;
  const from = hexRgb(BAR_COLORS[index]);
  const to = hexRgb(BAR_COLORS[index + 1]);
  const channel = (a, b) => Math.round(a + (b - a) * frac);
  return `rgb(${channel(from[0], to[0])}, ${channel(from[1], to[1])}, ${channel(from[2], to[2])})`;
}

function hexRgb(hex) {
  return [1, 3, 5].map((start) => parseInt(hex.slice(start, start + 2), 16));
}

function traceRoundRect(ctx, x, y, w, h) {
  const radius = Math.min(w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function drawBars() {
  const canvas = els.bars;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const gap = width * 0.014;
  const barWidth = (width - gap * (BAR_COUNT + 1)) / BAR_COUNT;
  const maxH = height * 0.9;
  for (let i = 0; i < BAR_COUNT; i += 1) {
    const barH = Math.max(0, state.heights[i]) * maxH;
    if (barH < 1) continue;
    const x = gap + i * (barWidth + gap);
    const y = (height - barH) / 2;
    ctx.shadowColor = mixColor(i / (BAR_COUNT - 1));
    ctx.shadowBlur = reducedMotion() ? 0 : height * 0.035;
    ctx.fillStyle = ctx.shadowColor;
    traceRoundRect(ctx, x, y, barWidth, barH);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
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
  for (let i = 0; i < BAR_COUNT; i += 1) {
    state.heights[i] += (targets[i] - state.heights[i]) * ease;
  }
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
  els.listen.addEventListener("click", onListenClick);
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  window.requestAnimationFrame(tick);
}

boot();
