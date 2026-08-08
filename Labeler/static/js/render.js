// render.js — all canvas drawing. #mel/#notes/#vel/#onset/#minimap/#ruler redraw on
// view/data changes (drawStage); #overlay redraws every rAF (drawOverlay).
import { store, pitchName, fmtTime, clamp, envDb, noteEnvCoeffs } from './state.js';
import {
  layout, timeToX, xToTime, binToY, binForPitch, pitchForBin,
  noteRect, visibleNotes, visibleTimeRange, durationS,
} from './timeline.js';
import { tileUrl } from './api.js';

const els = {};   // canvas elements
const cx = {};    // 2d contexts
let onTileLoad = () => {};

const OPEN_STRINGS = [
  { midi: 55, name: 'G3' }, { midi: 62, name: 'D4' },
  { midi: 69, name: 'A4' }, { midi: 76, name: 'E5' },
];

export function init(tileLoadCb){
  onTileLoad = tileLoadCb || (() => {});
  for (const id of ['ruler', 'mel', 'notes', 'vel', 'onset', 'overlay', 'minimap']){
    const el = document.getElementById(id);
    els[id] = el; cx[id] = el.getContext('2d');
  }
}

// ---------- sizing / layout ----------
export function resize(){
  const dpr = window.devicePixelRatio || 1;
  layout.dpr = dpr;
  const roll = document.getElementById('roll');
  const rollRect = roll.getBoundingClientRect();

  function size(id){
    const el = els[id];
    const r = el.getBoundingClientRect();
    el.width = Math.max(1, Math.round(r.width * dpr));
    el.height = Math.max(1, Math.round(r.height * dpr));
    cx[id].setTransform(dpr, 0, 0, dpr, 0, 0);
    return r;
  }
  const rRuler = size('ruler');
  const rMel = size('mel'); size('notes');
  const rVel = size('vel');
  const rOnset = size('onset');
  size('overlay'); size('minimap');

  layout.width = rMel.width;
  layout.rulerTop = rRuler.top - rollRect.top; layout.rulerH = rRuler.height;
  layout.rollTop = rMel.top - rollRect.top;    layout.rollH = rMel.height;
  layout.velTop = rVel.top - rollRect.top;      layout.velH = rVel.height;
  layout.onsetTop = rOnset.top - rollRect.top;  layout.onsetH = rOnset.height;
  layout.totalH = rollRect.height;
  const mm = els.minimap.getBoundingClientRect();
  layout.mmWidth = mm.width; layout.mmHeight = mm.height;
}

const crisp = (v) => Math.round(v) + 0.5;

// ---------- technique colors ----------
let colorMap = null;
function techColor(name){
  if (!colorMap){
    colorMap = new Map();
    for (const a of (store.config?.articulations || [])) colorMap.set(a.name, a.color);
  }
  return colorMap.get(name) || '#8a94a3';
}
export function resetColors(){ colorMap = null; }

// ---------- mel tile LRU ----------
const TILE_CAP = 30;
const tileCache = new Map();  // url -> {img, loaded}
function getTile(url){
  if (!url) return null;
  let t = tileCache.get(url);
  if (t){ tileCache.delete(url); tileCache.set(url, t); return t; }   // bump LRU
  const img = new Image();
  t = { img, loaded: false };
  img.onload = () => { t.loaded = true; onTileLoad(); };
  img.onerror = () => { t.loaded = false; };
  img.src = url;
  tileCache.set(url, t);
  while (tileCache.size > TILE_CAP){ const k = tileCache.keys().next().value; tileCache.delete(k); }
  return t;
}

function melFrameDur(){
  const c = store.config || {};
  return (c.hop || 512) / (c.sr || 44100);
}

// ---------- public draw entry ----------
export function drawStage(){
  drawRuler();
  drawMel();
  drawNotes();
  drawEnv();
  drawOnset();
  drawMinimap();
}

// ---------- ruler ----------
function drawRuler(){
  const g = cx.ruler, w = layout.width, h = layout.rulerH;
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#12151a'; g.fillRect(0, 0, w, h);
  const [t0, t1] = visibleTimeRange();
  const pps = store.view.pxPerSec;
  // pick a tick step giving ~80px spacing
  const targets = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60];
  let step = targets[targets.length - 1];
  for (const s of targets){ if (s * pps >= 60){ step = s; break; } }
  g.strokeStyle = '#2a3038'; g.fillStyle = '#8a94a3'; g.lineWidth = 1;
  g.font = '10px ui-monospace,monospace'; g.textBaseline = 'middle';
  const first = Math.ceil(t0 / step) * step;
  for (let t = first; t <= t1; t += step){
    const x = crisp(timeToX(t));
    g.beginPath(); g.moveTo(x, h - 7); g.lineTo(x, h); g.stroke();
    g.fillText(fmtTime(t), x + 3, h / 2 - 1);
  }
}

// ---------- mel spectrogram ----------
function drawMel(){
  const g = cx.mel, w = layout.width, h = layout.rollH;
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
  const meta = store.meta;
  if (!meta || !meta.mel){ return; }
  const fd = melFrameDur();
  const tf = meta.mel.tile_frames || 2048;
  const nTiles = meta.mel.n_tiles || 1;
  const nFrames = meta.mel.n_frames || (nTiles * tf);
  const widths = meta.mel.tile_widths;   // per-tile actual frame count (last is ragged)
  const tileDur = tf * fd;
  const [t0, t1] = visibleTimeRange();
  const first = Math.max(0, Math.floor(t0 / tileDur));
  const last = Math.min(nTiles - 1, Math.floor(t1 / tileDur));
  for (let i = first; i <= last; i++){
    const framesInTile = widths ? widths[i] : Math.min(tf, nFrames - i * tf);
    const leftT = i * tileDur;
    const contentR = leftT + framesInTile * fd;             // tile content end (ragged-aware)
    const visL = Math.max(leftT, t0), visR = Math.min(contentR, t1);
    if (visR <= visL) continue;
    const dx = timeToX(visL), dw = (visR - visL) * store.view.pxPerSec;
    const t = getTile(tileUrl(meta, i));
    if (!t || !t.loaded){
      g.fillStyle = 'rgba(30,34,42,.6)'; g.fillRect(dx, 0, dw, h);   // placeholder
      continue;
    }
    const img = t.img;
    // imsave writes one image pixel per mel frame, so frame index == image px (1:1).
    const sx = (visL - leftT) / fd;
    const sw = (visR - visL) / fd;
    g.imageSmoothingEnabled = true;
    g.drawImage(img, sx, 0, sw, img.naturalHeight, dx, 0, dw, h);
  }
  drawGridlines(g);
  drawMuteShading(g, h);
}

function drawGridlines(g){
  g.lineWidth = 1; g.font = '9px ui-sans-serif,system-ui';
  for (const s of OPEN_STRINGS){
    const y = crisp(binToY(binForPitch(s.midi)));
    g.strokeStyle = 'rgba(123,224,192,.22)';
    g.beginPath(); g.moveTo(0, y); g.lineTo(layout.width, y); g.stroke();
    g.fillStyle = 'rgba(123,224,192,.5)';
    g.fillText(s.name, 3, y - 3);
  }
}

function drawMuteShading(g, h){
  const regions = (store.doc && store.doc.mute_regions) || [];
  for (const r of regions){
    const x = timeToX(r.start_s), x2 = timeToX(r.end_s);
    const w = x2 - x;
    g.save();
    g.fillStyle = 'rgba(242,104,92,.14)'; g.fillRect(x, 0, w, h);
    g.strokeStyle = 'rgba(242,104,92,.5)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(crisp(x), 0); g.lineTo(crisp(x), h);
    g.moveTo(crisp(x2), 0); g.lineTo(crisp(x2), h); g.stroke();
    // hatch
    g.strokeStyle = 'rgba(242,104,92,.18)';
    for (let hx = x; hx < x2; hx += 8){ g.beginPath(); g.moveTo(hx, 0); g.lineTo(hx + h, h); g.stroke(); }
    g.restore();
  }
}

// ---------- notes ----------
function drawNotes(){
  const g = cx.notes, w = layout.width, h = layout.rollH;
  g.clearRect(0, 0, w, h);
  const draggingId = store.drag && (store.drag.kind === 'move' || store.drag.kind === 'resize') ? store.drag.id : null;
  for (const n of visibleNotes()){
    if (n.id === draggingId) continue;   // ghost drawn on overlay
    drawOneNote(g, n, n.id === store.selectedId);
  }
}

function drawOneNote(g, n, selected){
  const r = noteRect(n);
  const col = techColor(n.technique);
  g.globalAlpha = 0.78; g.fillStyle = col;
  g.fillRect(r.x, r.y, r.w, r.h);
  g.globalAlpha = 1;
  if (n.vibrato) drawVibrato(g, r, col, n);
  const human = n.human && (n.human.technique || n.human.vibrato || n.human.velocity || n.human.timing);
  if (human){
    g.fillStyle = '#fff';
    g.beginPath(); g.moveTo(r.x + r.w, r.y); g.lineTo(r.x + r.w - 4, r.y); g.lineTo(r.x + r.w, r.y + 4); g.fill();
  }
  if (selected){
    g.strokeStyle = '#fff'; g.lineWidth = 2;
    g.strokeRect(Math.round(r.x) + 1, Math.round(r.y) + 1, Math.round(r.w) - 2, Math.round(r.h) - 2);
  }
}

// Which vib_params slot holds the depth / rate, per model (common/vibrato_model.py).
// A model missing from this table draws the generic squiggle.
const VIB_PARAM_IX = { rampboth5: { depth: 0, rate: 2 } };

// Decorative squiggle above a vibrato note. With measured vib_params it is scaled:
// spatial frequency from the fitted rate (converted through this note's px/s so the
// wiggle count matches the real cycle count) and amplitude from the fitted half-depth.
// Deliberately cheap — the initial-depth/rate snapshot only, no ramps, no phase.
function drawVibrato(g, r, col, n){
  const ix = n && n.vib_params && VIB_PARAM_IX[n.vib_model];
  let k = 1 / 3, amp = 1.6;          // legacy fixed squiggle
  if (ix){
    const dur = Math.max(1e-3, n.end_s - n.start_s);
    const pxPerSec = Math.max(1e-3, r.w / dur);
    k = clamp(2 * Math.PI * n.vib_params[ix.rate] / pxPerSec, 0.05, 1.0);
    amp = clamp(n.vib_params[ix.depth] / 9.4, 0.6, Math.max(1, r.h * 0.4));
  }
  const y = r.y - 3, x0 = r.x, x1 = r.x + r.w;
  g.strokeStyle = col; g.lineWidth = 1.2; g.beginPath();
  for (let x = x0; x <= x1; x += 2){
    const yy = y - Math.sin((x - x0) * k) * amp;
    if (x === x0) g.moveTo(x, yy); else g.lineTo(x, yy);
  }
  g.stroke();
}

// ---------- envelope lane ----------
// Per-note A-weighted loudness envelope, rasterized as a filled dB curve.
// db(u), env_dct, and the velocity fallback (VEL_C0_A/VEL_C0_B) all live in state.js
// so render + synth share one definition (envDb / noteEnvCoeffs).
// Fixed display range (power-dB); values outside are clamped.
const ENV_DB_MIN = -30, ENV_DB_MAX = 16;
function envDbToY(db, h){
  const f = clamp((db - ENV_DB_MIN) / (ENV_DB_MAX - ENV_DB_MIN), 0, 1);
  return h - f * (h - 4);   // keep the old 4px top margin the vel bars had
}
// trace the top curve across [x0, x0+bw] with lineTo (path must be started by caller)
function traceEnvTop(g, c, x0, bw, h){
  for (let px = 0; px <= bw; px += 2) g.lineTo(x0 + px, envDbToY(envDb(c, bw > 0 ? px / bw : 0), h));
  g.lineTo(x0 + bw, envDbToY(envDb(c, 1), h));   // exact endpoint
}
function drawEnv(){
  const g = cx.vel, w = layout.width, h = layout.velH;
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#0b0d10'; g.fillRect(0, 0, w, h);
  for (const n of visibleNotes()){
    const hasEnv = Array.isArray(n.env_dct) && n.env_dct.length >= 3;
    const c = noteEnvCoeffs(n);
    const x0 = timeToX(n.start_s);
    const bw = Math.max(2, (n.end_s - n.start_s) * store.view.pxPerSec);
    const selected = n.id === store.selectedId;
    const col = techColor(n.technique);
    // filled region from lane bottom up to the curve (match vel-bar fill/alpha)
    g.globalAlpha = (selected ? 1 : 0.75) * (hasEnv ? 1 : 0.5);   // dim velocity-fallback notes
    g.fillStyle = col;
    g.beginPath();
    g.moveTo(x0, h);
    traceEnvTop(g, c, x0, bw, h);
    g.lineTo(x0 + bw, h);
    g.closePath();
    g.fill();
    g.globalAlpha = 1;
    // top edge: white when selected (mirrors old highlight); dashed when on the
    // velocity fallback so real-envelope notes are visually distinguishable.
    if (selected || !hasEnv){
      g.strokeStyle = selected ? '#fff' : col;
      g.lineWidth = 1;
      if (!hasEnv && !selected) g.setLineDash([3, 2]);
      g.beginPath();
      g.moveTo(x0, envDbToY(envDb(c, 0), h));
      traceEnvTop(g, c, x0, bw, h);
      g.stroke();
      g.setLineDash([]);
    }
  }
  g.fillStyle = '#6b7482'; g.font = '9px ui-sans-serif'; g.fillText('env', 4, 10);
}

// ---------- onset lane ----------
let onsetRef = null, onsetMax = 1;
function drawOnset(){
  const g = cx.onset, w = layout.width, h = layout.onsetH;
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#0b0d10'; g.fillRect(0, 0, w, h);
  const env = store.onset;
  if (!env || !env.length){ return; }
  if (env !== onsetRef){ onsetRef = env; onsetMax = 0; for (let i = 0; i < env.length; i++) onsetMax = Math.max(onsetMax, env[i]); onsetMax = onsetMax || 1; }
  const fd = melFrameDur();
  const [t0, t1] = visibleTimeRange();
  let f0 = Math.max(0, Math.floor(t0 / fd)), f1 = Math.min(env.length - 1, Math.ceil(t1 / fd));
  g.strokeStyle = '#5b9dff'; g.lineWidth = 1; g.beginPath();
  let started = false;
  for (let f = f0; f <= f1; f++){
    const x = timeToX(f * fd);
    const y = h - (env[f] / onsetMax) * (h - 3);
    if (!started){ g.moveTo(x, y); started = true; } else g.lineTo(x, y);
  }
  g.stroke();
  g.fillStyle = '#6b7482'; g.font = '9px ui-sans-serif'; g.fillText('onset', 4, 10);
}

// ---------- minimap ----------
function drawMinimap(){
  const g = cx.minimap, w = layout.mmWidth, h = layout.mmHeight;
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#12151a'; g.fillRect(0, 0, w, h);
  const dur = durationS() || 1;
  const notes = (store.doc && store.doc.notes) || [];
  let pmin = 127, pmax = 0;
  for (const n of notes){ pmin = Math.min(pmin, n.pitch); pmax = Math.max(pmax, n.pitch); }
  if (pmax < pmin){ pmin = 55; pmax = 88; }
  const span = Math.max(1, pmax - pmin);
  for (const n of notes){
    const x = (n.start_s / dur) * w;
    const x2 = (n.end_s / dur) * w;
    const y = h - 3 - ((n.pitch - pmin) / span) * (h - 8);
    g.strokeStyle = techColor(n.technique); g.globalAlpha = 0.85; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(x, y); g.lineTo(Math.max(x + 1, x2), y); g.stroke();
  }
  g.globalAlpha = 1;
  // viewport rect
  const [t0, t1] = visibleTimeRange();
  const vx = (t0 / dur) * w, vw = ((t1 - t0) / dur) * w;
  g.fillStyle = 'rgba(91,157,255,.14)'; g.fillRect(vx, 0, vw, h);
  g.strokeStyle = 'rgba(91,157,255,.8)'; g.lineWidth = 1; g.strokeRect(crisp(vx), .5, vw, h - 1);
  // playhead
  const px = (store.playhead / dur) * w;
  g.strokeStyle = '#e7ecf2'; g.beginPath(); g.moveTo(crisp(px), 0); g.lineTo(crisp(px), h); g.stroke();
}

// ---------- overlay (rAF) ----------
export function drawOverlay(){
  const g = cx.overlay, w = layout.width, H = layout.totalH;
  g.clearRect(0, 0, w, H);

  // drag ghosts
  if (store.drag) drawDragGhost(g);

  // hover crosshair (roll band)
  if (store.hover && store.hover.bin != null){
    const hx = crisp(timeToX(store.hover.t));
    const hy = crisp(layout.rollTop + binToY(store.hover.bin));
    g.strokeStyle = 'rgba(231,236,242,.22)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, hy); g.lineTo(w, hy); g.stroke();
    g.fillStyle = 'rgba(231,236,242,.85)'; g.font = '10px ui-monospace,monospace';
    const label = `${fmtTime(store.hover.t)}  ${pitchName(pitchForBin(store.hover.bin))}`;
    const tw = g.measureText(label).width + 8;
    let bx = timeToX(store.hover.t) + 8; if (bx + tw > w) bx = w - tw - 2;
    g.fillStyle = 'rgba(10,12,16,.85)'; g.fillRect(bx, hy - 16, tw, 13);
    g.fillStyle = 'rgba(231,236,242,.9)'; g.fillText(label, bx + 4, hy - 9);
  }

  // playhead across everything
  const x = crisp(timeToX(store.playhead));
  g.strokeStyle = '#e7ecf2'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
  g.fillStyle = '#e7ecf2';
  g.beginPath(); g.moveTo(x - 4, 0); g.lineTo(x + 4, 0); g.lineTo(x, 6); g.fill();
}

function drawDragGhost(g){
  const d = store.drag;
  if (d.kind === 'region'){
    const x = timeToX(Math.min(d.start, d.cur)), x2 = timeToX(Math.max(d.start, d.cur));
    g.fillStyle = 'rgba(242,104,92,.2)'; g.fillRect(x, 0, x2 - x, layout.totalH);
    g.strokeStyle = 'rgba(242,104,92,.8)'; g.lineWidth = 1; g.strokeRect(crisp(x), .5, x2 - x, layout.totalH - 1);
    return;
  }
  if (d.kind === 'move' || d.kind === 'resize' || d.kind === 'create'){
    const r = d.previewRect;
    if (!r) return;
    const y = r.y + layout.rollTop;
    g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.setLineDash([4, 3]);
    g.strokeRect(r.x + .5, y + .5, r.w, r.h); g.setLineDash([]);
    if (d.pitchLabel){
      g.fillStyle = 'rgba(10,12,16,.9)'; g.fillRect(r.x, y - 15, 30, 13);
      g.fillStyle = '#7ee0c0'; g.font = '10px ui-monospace,monospace'; g.fillText(d.pitchLabel, r.x + 3, y - 8);
    }
  }
}
