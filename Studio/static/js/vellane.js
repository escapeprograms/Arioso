// vellane.js — the bottom control lane. Owns four switchable modes (Velocity / Envelope /
// Pan / Pitch), their canvas rendering, the left-column mode selector, and drag editing.
// Bipolar modes (Pan, Pitch) draw bars from a center zero-line; Velocity is bottom-anchored.
// Pitch writes a single flat control point per note ([{beat:0, semitones:v}]) and is
// read-only on notes that already carry a multi-point bend curve (edit those in the bend
// editor). Envelope mode edits per-note dB energy curves via three handles (start/mid/end)
// with mass delta-drag across the whole selection plus a horizontal 2-axis mid handle on a
// lone selected note; each commit is one undoable command (setVelocities / setEnvs /
// setPans / setBendsFlat). Cross-mode dotted overlays show the other view for context:
// the envelope curve in Velocity mode, the velocity level in Envelope mode.
import {
  store, apply, isSelected, selectOnly, clamp, timeSig, notify, noteById,
  noteEnvCoeffs, envDb, cpFromCoeffs, coeffsFromCp, c0FromVel, coeffsThroughPoint, envMidU,
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
const MODES = ['velocity', 'envelope', 'pan', 'pitch'];
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

  // cross-mode context (velocity mode): dotted envelope-curve overlay on notes that carry
  // env_dct (env-less notes skipped — their flat fallback would just trace the bar top).
  if (!bipolar){
    g.save(); g.setLineDash([2, 3]); g.globalAlpha = 0.35; g.lineWidth = 1;
    for (const n of visibleNotes()){
      if (!(Array.isArray(n.env_dct) && n.env_dct.length >= 3)) continue;
      const x0 = Math.round(beatToX(n.start_beat));
      const bw = Math.max(2, Math.round(n.len_beats * pxb));
      g.strokeStyle = isSelected(n.id) ? '#fff' : shade(techColor(n.technique), 0.3);
      strokeEnvCurve(g, noteEnvCoeffs(n), x0, bw, h);
    }
    g.restore();
  }
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
// dotted top-curve overlay for cross-mode context (display-only)
function strokeEnvCurve(g, c, x0, bw, h){
  g.beginPath();
  g.moveTo(x0, envDbToY(envDb(c, 0), h));
  traceEnvTop(g, c, x0, bw, h);
  g.stroke();
}
// flat dotted line at the velocity-equivalent dB level across the note span (display-only)
function strokeVelLevel(g, n, x0, bw, h){
  const y = envDbToY(c0FromVel(n.velocity), h);
  g.beginPath(); g.moveTo(x0, y); g.lineTo(x0 + bw, y); g.stroke();
}
// every selected visible note carries live handles in envelope mode
function selectedEnvNotes(){ return visibleNotes().filter(n => isSelected(n.id)); }
// handle u-positions: start(0), interior extremum (arch peak), end(1) — so the mid handle
// tracks the curve's peak after a horizontal drag. Used for BOTH drawing and hit-testing.
const handleUs = (c) => [0, envMidU(c), 1];
// closest handle across ALL selected visible notes within HITR; { note, idx, u, db0 } or null
function hitHandleAny(x, y, h){
  let best = null, bestD = HITR * HITR;
  for (const n of selectedEnvNotes()){
    const c = noteEnvCoeffs(n);
    const x0 = Math.round(beatToX(n.start_beat));
    const bw = Math.max(2, Math.round(n.len_beats * store.view.px_per_beat));
    const us = handleUs(c);
    for (let i = 0; i < 3; i++){
      const u = us[i];
      const hx = x0 + bw * u, hy = envDbToY(envDb(c, u), h);
      const d = (hx - x) * (hx - x) + (hy - y) * (hy - y);
      if (d <= bestD){ bestD = d; best = { note: n, idx: i, u, db0: envDb(c, u) }; }
    }
  }
  return best;
}
function drawEnvelope(g, w, h){
  // preview coeffs map during a live drag (aliases the in-place drag.preview arrays)
  const preview = (store.drag && store.drag.kind === 'lane' && store.drag.mode === 'envelope') ? store.drag.preview : null;

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
      strokeEnvCurve(g, c, x0, bw, h);
      g.setLineDash([]);
    }
  }

  // cross-mode context: dotted flat velocity-level line across every note (display-only)
  g.save(); g.setLineDash([2, 3]); g.globalAlpha = 0.35; g.lineWidth = 1;
  for (const n of visibleNotes()){
    const x0 = Math.round(beatToX(n.start_beat));
    const bw = Math.max(2, Math.round(n.len_beats * store.view.px_per_beat));
    g.strokeStyle = isSelected(n.id) ? '#fff' : shade(techColor(n.technique), 0.3);
    strokeVelLevel(g, n, x0, bw, h);
  }
  g.restore();

  // handles (start / arch-peak / end) on EVERY selected visible note; the actively dragged
  // one is drawn larger + white. During a single-note 2-axis mid drag the mid handle follows
  // the pointer (drag.u) — the curve passes through that point, but its extremum may sit
  // elsewhere, so the handle can shift slightly to the extremum on release.
  const dragging = drag && drag.mode === 'envelope';
  for (const n of selectedEnvNotes()){
    let c = noteEnvCoeffs(n);
    if (preview && preview[n.id]) c = preview[n.id];
    const x0 = Math.round(beatToX(n.start_beat));
    const bw = Math.max(2, Math.round(n.len_beats * store.view.px_per_beat));
    const isAnchor = dragging && drag.anchorId === n.id;
    const us = handleUs(c);
    for (let i = 0; i < 3; i++){
      let u = us[i];
      if (isAnchor && i === 1 && drag.single && drag.pointIdx === 1) u = drag.u;
      const isDrag = isAnchor && i === drag.pointIdx;
      const hx = x0 + bw * u, hy = envDbToY(envDb(c, u), h);
      g.beginPath(); g.arc(hx, hy, isDrag ? 5 : 3, 0, Math.PI * 2);
      g.fillStyle = isDrag ? '#fff' : '#ffcf6b'; g.fill();
      g.strokeStyle = 'rgba(0,0,0,0.7)'; g.lineWidth = 1; g.stroke();
    }
  }

  // dB value label near the dragged handle (bendedit style); mid-drag also shows position
  if (dragging){
    const n = noteById(drag.anchorId);
    if (n){
      let c = (preview && preview[n.id]) ? preview[n.id] : noteEnvCoeffs(n);
      const x0 = Math.round(beatToX(n.start_beat));
      const bw = Math.max(2, Math.round(n.len_beats * store.view.px_per_beat));
      const midDrag = drag.pointIdx === 1 && drag.single;
      const u = midDrag ? drag.u : handleUs(c)[drag.pointIdx];
      const hx = x0 + bw * u, hy = envDbToY(envDb(c, u), h);
      const db = envDb(c, u);
      const txt = (db >= 0 ? '+' : '') + db.toFixed(2) + ' dB' + (midDrag ? ' @ ' + Math.round(u * 100) + '%' : '');
      const wLabel = midDrag ? 92 : 58;
      g.fillStyle = 'rgba(0,0,0,0.8)'; g.fillRect(hx + 6, hy - 14, wLabel, 13);
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
  const hit = hitHandleAny(x, y, h);
  if (hit){
    // snapshot the WHOLE selection (not just visible notes) — a mass drag must move
    // selected notes that are scrolled out of view too; hit-testing stays visible-only.
    const base = {}, preview = {};
    for (const id of store.selection){
      const n = noteById(id); if (!n) continue;
      base[id] = [...noteEnvCoeffs(n)]; preview[id] = [...noteEnvCoeffs(n)];
    }
    drag = {
      mode: 'envelope', anchorId: hit.note.id, pointIdx: hit.idx,
      db0: hit.db0, u0: hit.u, u: hit.u, single: store.selection.size === 1, base, preview,
    };
    // store.drag.preview MUST alias the same live arrays draw() reads (never reassign them)
    store.drag = { kind: 'lane', mode: 'envelope', preview };
    requestStatic();
    return;
  }
  const n = noteUnderBeat(xToBeat(x));
  if (n) selectOnly(n.id);   // no drag armed; handles move to the newly selected note
  requestStatic();
}
function onMove(e){
  if (!drag) return;
  const { x, y, h } = xy(e);
  if (drag.mode === 'envelope'){
    const dbNew = clamp(Math.round(envDbFromY(y, h) / ENV_ROUND) * ENV_ROUND, ENV_DB_MIN, ENV_DB_MAX);
    if (drag.pointIdx === 1 && drag.single){
      // single note, mid handle: 2-axis drag — vertical dB + horizontal peak position um.
      const n = noteById(drag.anchorId);
      const x0 = Math.round(beatToX(n.start_beat));
      const bw = Math.max(2, Math.round(n.len_beats * store.view.px_per_beat));
      const um = clamp((x - x0) / bw, 0.1, 0.9);
      const cp = cpFromCoeffs(drag.base[n.id]);
      const next = coeffsThroughPoint(cp.start, cp.end, um, dbNew);
      drag.u = um;
      const p = drag.preview[n.id];
      p[0] = next[0]; p[1] = next[1]; p[2] = next[2];   // mutate IN PLACE
    } else {
      // start/end handle, or any handle in a multi-selection: vertical delta-drag.
      const delta = dbNew - drag.db0;
      const key = ['start', 'mid', 'end'][drag.pointIdx];
      for (const id in drag.base){
        const cp = cpFromCoeffs(drag.base[id]);
        cp[key] = clamp(cp[key] + delta, ENV_DB_MIN, ENV_DB_MAX);   // clamp each note's point independently
        const next = coeffsFromCp(cp.start, cp.mid, cp.end);
        const p = drag.preview[id];
        p[0] = next[0]; p[1] = next[1]; p[2] = next[2];   // mutate IN PLACE
      }
    }
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
    // commit only notes whose coeffs actually changed — a no-move click must not stamp
    // env_dct onto env-less notes (segment-hash stability) or push a no-op undo entry.
    const changed = {};
    for (const id in d.preview){
      const a = d.preview[id], b = d.base[id];
      if (a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2]) changed[id] = a;
    }
    if (Object.keys(changed).length) apply(edit.setEnvs(changed, [...store.selection]));
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
