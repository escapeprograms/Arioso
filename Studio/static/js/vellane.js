// vellane.js — the bottom control lane. Owns three switchable modes (Velocity / Pan /
// Pitch), their canvas rendering, the left-column mode selector, and drag-paint editing.
// Bipolar modes (Pan, Pitch) draw bars from a center zero-line; Velocity is bottom-anchored.
// Pitch writes a single flat control point per note ([{beat:0, semitones:v}]) and is
// read-only on notes that already carry a multi-point bend curve (edit those in the bend
// editor). Every commit is one undoable command (setVelocities / setPans / setBendsFlat).
import {
  store, apply, isSelected, selectOnly, clamp, timeSig, notify,
} from './state.js';
import { beatToX, xToBeat, visibleNotes, visibleBeatRange } from './timeline.js';
import * as edit from './editing.js';
import { techColor, shade, mix, SELECT_COL } from './render.js';

const LANE_BG = '#1b2228';
const PAD = 6;

let requestStatic = () => {};

// ---------- mode state ----------
const MODES = ['velocity', 'pan', 'pitch'];
export function setMode(m){
  if (!MODES.includes(m) || store.lane === m) { syncButtons(); return; }
  store.lane = m;
  syncButtons();
  requestStatic();
  notify();
}
function syncButtons(){
  document.querySelectorAll('#lane-picker [data-lane]').forEach(b => {
    b.classList.toggle('on', b.dataset.lane === store.lane);
  });
}

export function init(deps = {}){
  requestStatic = deps.requestStatic || (() => {});
  document.querySelectorAll('#lane-picker [data-lane]').forEach(b => {
    b.onclick = () => setMode(b.dataset.lane);
  });
  syncButtons();
  const lane = document.getElementById('lane');
  lane.addEventListener('pointerdown', onDown);
  lane.addEventListener('pointermove', onMove);
  lane.addEventListener('pointerup', onUp);
  lane.addEventListener('pointercancel', onUp);
  lane.addEventListener('contextmenu', (e) => e.preventDefault());
}

// ---------- per-note current value in the active mode ----------
function curVal(n, mode){
  if (mode === 'pan') return n.pan || 0;
  if (mode === 'pitch'){ const b = n.bend || []; return b.length ? b[0].semitones : 0; }
  return n.velocity;
}
function isReadonly(n, mode){ return mode === 'pitch' && (n.bend || []).length > 1; }

const modeMax = { pan: 1, pitch: 2 };   // bipolar extents (velocity handled separately)

// ---------- drawing ----------
export function draw(g, w, h){
  g.clearRect(0, 0, w, h);
  g.fillStyle = LANE_BG; g.fillRect(0, 0, w, h);

  const mode = store.lane;
  const bipolar = mode !== 'velocity';
  const cy = Math.round(h / 2);
  const [b0, b1] = visibleBeatRange();
  const num = timeSig().num || 4;
  const pxb = store.view.px_per_beat;

  // bar-line orientation marks
  g.strokeStyle = 'rgba(255,255,255,0.05)'; g.lineWidth = 1; g.beginPath();
  const firstBar = Math.ceil(b0 / num) * num;
  for (let beat = firstBar; beat <= b1; beat += num){ const x = Math.round(beatToX(beat)) + 0.5; g.moveTo(x, 0); g.lineTo(x, h); }
  g.stroke();

  // zero / center line for bipolar modes
  if (bipolar){
    g.strokeStyle = 'rgba(255,255,255,0.18)'; g.lineWidth = 1;
    g.setLineDash([3, 3]); g.beginPath(); g.moveTo(0, cy + 0.5); g.lineTo(w, cy + 0.5); g.stroke(); g.setLineDash([]);
  }

  const preview = (store.drag && store.drag.kind === 'lane' && store.drag.mode === mode) ? store.drag.preview : null;
  const half = cy - PAD;

  for (const n of visibleNotes()){
    const x = Math.round(beatToX(n.start_beat));
    const bw = Math.max(2, Math.round(n.len_beats * pxb));
    let val = curVal(n, mode);
    if (preview && preview[n.id] != null) val = preview[n.id];
    const sel = isSelected(n.id);
    const readonly = isReadonly(n, mode);
    const base = techColor(n.technique);
    const col = sel ? mix(base, SELECT_COL, 0.6) : base;

    if (!bipolar){
      const frac = clamp(val / 127, 0, 1);
      const top = Math.round(h - frac * (h - PAD));
      g.globalAlpha = sel ? 1 : 0.82; g.fillStyle = col;
      g.fillRect(x, top, bw, h - top);
      g.globalAlpha = 1;
      g.beginPath(); g.arc(x + bw / 2, top, 2.6, 0, Math.PI * 2);
      g.fillStyle = sel ? '#fff' : shade(col, 0.3); g.fill();
    } else {
      const mx = modeMax[mode];
      const frac = clamp(val / mx, -1, 1);
      const tip = Math.round(cy - frac * half);
      g.globalAlpha = readonly ? 0.3 : (sel ? 1 : 0.82);
      g.fillStyle = col;
      if (tip <= cy) g.fillRect(x, tip, bw, cy - tip);
      else g.fillRect(x, cy, bw, tip - cy);
      g.globalAlpha = 1;
      g.beginPath(); g.arc(x + bw / 2, tip, 2.6, 0, Math.PI * 2);
      g.fillStyle = readonly ? '#8a8a8a' : (sel ? '#fff' : shade(col, 0.3)); g.fill();
      if (readonly){   // marker: this note's pitch is a multi-point curve (edit in bend editor)
        g.strokeStyle = 'rgba(255,255,255,0.5)'; g.lineWidth = 1;
        const my = tip - 6, mx0 = x + bw / 2 - 3;
        g.beginPath(); g.moveTo(mx0, my + 2); g.lineTo(mx0 + 2, my - 1); g.lineTo(mx0 + 4, my + 2); g.lineTo(mx0 + 6, my - 1); g.stroke();
      }
    }
  }
  g.globalAlpha = 1;
}

// ---------- value <-> y ----------
function valueFromY(y, mode, h){
  if (mode === 'velocity') return clamp(Math.round((1 - y / h) * 127), 1, 127);
  const cy = h / 2, half = cy - PAD, mx = modeMax[mode];
  const raw = ((cy - y) / half) * mx;
  return clamp(Math.round(raw * 100) / 100, -mx, mx);
}
function noteUnderBeat(beat){
  let best = null, bestw = Infinity;
  for (const n of visibleNotes()){
    if (beat >= n.start_beat && beat <= n.start_beat + n.len_beats && n.len_beats < bestw){ bestw = n.len_beats; best = n; }
  }
  return best;
}

// ---------- drag-paint ----------
let drag = null;
function xy(e){ const r = document.getElementById('lane').getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top, h: r.height }; }

function paintAt(x, y, h, select){
  const mode = store.lane;
  const n = noteUnderBeat(xToBeat(x));
  if (!n || isReadonly(n, mode)) return;
  if (select) selectOnly(n.id);
  drag.map[n.id] = valueFromY(y, mode, h);
}
function onDown(e){
  if (e.button !== 0) return;
  const lane = document.getElementById('lane');
  lane.setPointerCapture(e.pointerId);
  const { x, y, h } = xy(e);
  drag = { map: {}, mode: store.lane };
  store.drag = { kind: 'lane', mode: store.lane, preview: drag.map, hideIds: new Set() };
  paintAt(x, y, h, true);   // select the first note touched
  requestStatic();
}
function onMove(e){
  if (!drag) return;
  const { x, y, h } = xy(e);
  paintAt(x, y, h, false);
  requestStatic();
}
function onUp(e){
  if (!drag) return;
  const { map, mode } = drag; drag = null;
  try { document.getElementById('lane').releasePointerCapture(e.pointerId); } catch {}
  store.drag = null;
  const ids = Object.keys(map);
  if (ids.length){
    if (mode === 'velocity') apply(edit.setVelocities(map));
    else if (mode === 'pan') apply(edit.setPans(map));
    else apply(edit.setBendsFlat(map));
  }
  requestStatic();
}
