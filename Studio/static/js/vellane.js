// vellane.js — the bottom control lane. Owns three switchable modes (Velocity / Pan /
// Pitch), their canvas rendering, the left-column mode selector, and drag-paint editing.
// Bipolar modes (Pan, Pitch) draw bars from a center zero-line; Velocity is bottom-anchored.
// Pitch writes a single flat control point per note ([{beat:0, semitones:v}]) and is
// read-only on notes that already carry a multi-point bend curve (edit those in the bend
// editor). Every commit is one undoable command (setVelocities / setPans / setBendsFlat).
import {
  store, apply, isSelected, selectOnly, clamp, timeSig, notify, noteById,
  noteEnvCoeffs, envDb, cpFromCoeffs, coeffsFromCp,
} from './state.js';
import { beatToX, xToBeat, visibleNotes, visibleBeatRange } from './timeline.js';
import * as edit from './editing.js';
import { techColor, shade, mix, SELECT_COL } from './render.js';

const LANE_BG = '#1b2228';
const PAD = 6;

// ---------- envelope-mode constants ----------
const ENV_DB_MIN = -30, ENV_DB_MAX = 16;   // fixed display range (mirrors Labeler env lane)
const HITR = 7;                            // handle pick radius (px)
const ENV_ROUND = 0.01;                    // dB rounding step while dragging

let requestStatic = () => {};

// ---------- mode state ----------
const MODES = ['velocity', 'pan', 'pitch', 'envelope'];
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

  // envelope mode owns its own rasterization (filled dB curves + handles); it never paints
  if (mode === 'envelope'){ drawEnvelope(g, w, h); return; }

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

// ---------- envelope mode drawing ----------
// dB <-> y with a PAD-px top margin (matches the velocity bars' top gap).
function envDbToY(db, h){
  const f = clamp((db - ENV_DB_MIN) / (ENV_DB_MAX - ENV_DB_MIN), 0, 1);
  return h - f * (h - PAD);
}
function envDbFromY(y, h){
  const f = clamp((h - y) / (h - PAD), 0, 1);
  return ENV_DB_MIN + f * (ENV_DB_MAX - ENV_DB_MIN);
}
// trace the top curve across [x0, x0+bw] with lineTo (path started by the caller)
function traceEnvTop(g, c, x0, bw, h){
  for (let px = 0; px <= bw; px += 2) g.lineTo(x0 + px, envDbToY(envDb(c, bw > 0 ? px / bw : 0), h));
  g.lineTo(x0 + bw, envDbToY(envDb(c, 1), h));   // exact endpoint at u=1
}
// the single note whose handles are live (envelope mode edits one selected note)
function envSelNote(){ return store.selection.size === 1 ? noteById([...store.selection][0]) : null; }
// hit-test the selected note's three handles; returns 0/1/2 or -1
function hitHandle(n, x, y, h){
  const c = noteEnvCoeffs(n);
  const x0 = Math.round(beatToX(n.start_beat));
  const bw = Math.max(2, Math.round(n.len_beats * store.view.px_per_beat));
  const us = [0, 0.5, 1];
  let best = -1, bestD = HITR * HITR;
  for (let i = 0; i < 3; i++){
    const hx = x0 + bw * us[i], hy = envDbToY(envDb(c, us[i]), h);
    const d = (hx - x) * (hx - x) + (hy - y) * (hy - y);
    if (d <= bestD){ bestD = d; best = i; }
  }
  return best;
}
function drawEnvelope(g, w, h){
  // preview coeffs during a live drag (aliases the in-place drag.coeffs array)
  const preview = (store.drag && store.drag.kind === 'lane' && store.drag.mode === 'envelope') ? store.drag.preview : null;
  const selNote = envSelNote();

  // soft gridlines: 0 dB and the corpus-median level (5.4 dB) as visual anchors.
  // (The velocity map now spans the full display range, so its edges ARE the lane edges.)
  g.lineWidth = 1;
  for (const [db, a] of [[0, 0.16], [5.4, 0.08]]){
    const y = Math.round(envDbToY(db, h)) + 0.5;
    g.strokeStyle = `rgba(255,255,255,${a})`;
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }

  for (const n of visibleNotes()){
    const hasEnv = Array.isArray(n.env_dct) && n.env_dct.length >= 3;
    let c = noteEnvCoeffs(n);
    if (preview && preview[n.id]) c = preview[n.id];
    const x0 = Math.round(beatToX(n.start_beat));
    const bw = Math.max(2, Math.round(n.len_beats * store.view.px_per_beat));
    const sel = isSelected(n.id);
    const base = techColor(n.technique);
    const col = sel ? mix(base, SELECT_COL, 0.6) : base;

    // filled region from lane bottom up to the curve; env-less notes dimmed to 0.5x
    g.globalAlpha = (sel ? 1 : 0.82) * (hasEnv ? 1 : 0.5);
    g.fillStyle = col;
    g.beginPath();
    g.moveTo(x0, h);
    traceEnvTop(g, c, x0, bw, h);
    g.lineTo(x0 + bw, h);
    g.closePath(); g.fill();
    g.globalAlpha = 1;

    // top edge: white when selected; dashed + dim when env-less (velocity fallback)
    if (sel || !hasEnv){
      g.strokeStyle = sel ? '#fff' : col;
      g.lineWidth = 1;
      if (!hasEnv && !sel) g.setLineDash([3, 2]);
      g.beginPath();
      g.moveTo(x0, envDbToY(envDb(c, 0), h));
      traceEnvTop(g, c, x0, bw, h);
      g.stroke();
      g.setLineDash([]);
    }
  }

  // handles (u=0/0.5/1) on the selected note only; dragged one larger + white
  if (selNote){
    let c = noteEnvCoeffs(selNote);
    if (preview && preview[selNote.id]) c = preview[selNote.id];
    const x0 = Math.round(beatToX(selNote.start_beat));
    const bw = Math.max(2, Math.round(selNote.len_beats * store.view.px_per_beat));
    const dragIdx = (drag && drag.mode === 'envelope' && drag.id === selNote.id) ? drag.pointIdx : -1;
    const us = [0, 0.5, 1];
    for (let i = 0; i < 3; i++){
      const hx = x0 + bw * us[i], hy = envDbToY(envDb(c, us[i]), h);
      g.beginPath(); g.arc(hx, hy, i === dragIdx ? 5 : 3.4, 0, Math.PI * 2);
      g.fillStyle = i === dragIdx ? '#fff' : '#ffcf6b'; g.fill();
      g.strokeStyle = 'rgba(0,0,0,0.7)'; g.lineWidth = 1; g.stroke();
    }
    // dB value label near the dragged handle (bendedit style)
    if (dragIdx >= 0){
      const u = us[dragIdx];
      const hx = x0 + bw * u, hy = envDbToY(envDb(c, u), h);
      const db = envDb(c, u);
      const txt = (db >= 0 ? '+' : '') + db.toFixed(2) + ' dB';
      g.fillStyle = 'rgba(0,0,0,0.8)'; g.fillRect(hx + 6, hy - 14, 58, 13);
      g.fillStyle = '#ffd479'; g.font = '10px ui-monospace,monospace'; g.textBaseline = 'middle';
      g.fillText(txt, hx + 9, hy - 7);
    }
  }
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
  if (e.button !== 0) return;   // right-click is a no-op in every mode
  const lane = document.getElementById('lane');
  lane.setPointerCapture(e.pointerId);
  const { x, y, h } = xy(e);
  if (store.lane === 'envelope'){ onDownEnv(x, y, h); return; }
  drag = { map: {}, mode: store.lane };
  store.drag = { kind: 'lane', mode: store.lane, preview: drag.map, hideIds: new Set() };
  paintAt(x, y, h, true);   // select the first note touched
  requestStatic();
}
// envelope mode NEVER paints: a handle hit arms a control-point drag; a miss click-selects
// the note under the cursor (so its handles become grabbable on the next pointerdown).
function onDownEnv(x, y, h){
  const selNote = envSelNote();
  if (selNote){
    const idx = hitHandle(selNote, x, y, h);
    if (idx >= 0){
      const coeffs = [...noteEnvCoeffs(selNote)];
      drag = { mode: 'envelope', id: selNote.id, pointIdx: idx, coeffs };
      // store.drag.preview[id] MUST stay the same live object as drag.coeffs (alias)
      store.drag = { kind: 'lane', mode: 'envelope', preview: { [selNote.id]: coeffs } };
      requestStatic();
      return;
    }
  }
  const n = noteUnderBeat(xToBeat(x));
  if (n) selectOnly(n.id);   // no drag armed; handles move to the newly selected note
  requestStatic();
}
function onMove(e){
  if (!drag) return;
  const { x, y, h } = xy(e);
  if (drag.mode === 'envelope'){
    const db = clamp(Math.round(envDbFromY(y, h) / ENV_ROUND) * ENV_ROUND, ENV_DB_MIN, ENV_DB_MAX);
    const cp = cpFromCoeffs(drag.coeffs);
    if (drag.pointIdx === 0) cp.start = db;
    else if (drag.pointIdx === 1) cp.mid = db;
    else cp.end = db;
    const next = coeffsFromCp(cp.start, cp.mid, cp.end);
    // mutate the previewed array IN PLACE (never reassign — draw reads this same object)
    drag.coeffs[0] = next[0]; drag.coeffs[1] = next[1]; drag.coeffs[2] = next[2];
    requestStatic();
    return;
  }
  paintAt(x, y, h, false);
  requestStatic();
}
function onUp(e){
  if (!drag) return;
  if (drag.mode === 'envelope'){
    const d = drag; drag = null;
    try { document.getElementById('lane').releasePointerCapture(e.pointerId); } catch {}
    store.drag = null;
    apply(edit.setEnv(d.id, d.coeffs));   // one atomic undoable command per gesture
    requestStatic();
    return;
  }
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
