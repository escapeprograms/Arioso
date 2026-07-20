// bendedit.js — per-note pitch-bend polyline editor drawn on #overlay. Opened by a
// double-click on a note (interact.js routes pointer + dblclick input here while open).
// Control points are {beat, semitones} relative to the note start; beat clamped to the
// note length, semitones clamped to ±2 (the prior's hard bend range). The first point is
// pinned at beat 0 and always exists. Gestures commit one undoable setBend each:
//   • drag a point            → move (beat clamped between neighbors, semitones ±2)
//   • double-click a segment  → add a point
//   • right-click a point     → remove it (never the first)
// Esc or a click outside the panel commits & closes.
import { noteById, selectOnly, clamp, apply } from './state.js';
import { layout, beatToX, xToBeat, pitchToY } from './timeline.js';
import * as edit from './editing.js';

const MAXST = 2;              // ±2 semitone clamp (matches the saw prior)
const SAT = MAXST - 0.02;     // saturation warning threshold
const HITR = 7;               // control-point pick radius (px)
const MINGAP = 1e-3;

let activeId = null;
let pts = [];                 // working copy: [{beat, semitones}] (beat rel. note start)
let dragIdx = -1;
let requestStatic = () => {};
let keyHandler = null;

export function isOpen(){ return activeId != null; }

// ---------- geometry (depends on the note's current rect + view) ----------
function note(){ return noteById(activeId); }
function halfSpan(){ return Math.max(18, 2.6 * layout.rowH); }   // px per +MAXST side
function centerY(n){ return pitchToY(n.pitch) + layout.rowH / 2; }
function semToY(n, s){ return centerY(n) - (s / MAXST) * halfSpan(); }
function yToSem(n, y){ return clamp(Math.round(((centerY(n) - y) / halfSpan()) * MAXST * 100) / 100, -MAXST, MAXST); }
function beatToXRel(n, bRel){ return beatToX(n.start_beat + bRel); }
function xToBeatRel(n, x){ return clamp(Math.round((xToBeat(x) - n.start_beat) * 1000) / 1000, 0, n.len_beats); }
function panel(n){
  const x0 = beatToXRel(n, 0), x1 = beatToXRel(n, n.len_beats);
  return { x0: Math.min(x0, x1) - 6, x1: Math.max(x0, x1) + 6, y0: semToY(n, MAXST) - 6, y1: semToY(n, -MAXST) + 6 };
}

// ---------- open / close ----------
export function open(n, deps = {}){
  requestStatic = deps.requestStatic || requestStatic;
  activeId = n.id;
  selectOnly(n.id);
  pts = clonePts(n.bend);
  dragIdx = -1;
  keyHandler = (e) => {
    if (e.key === 'Escape'){ e.stopImmediatePropagation(); e.preventDefault(); close(); }
  };
  window.addEventListener('keydown', keyHandler, true);
  requestStatic();
}
export function close(){
  if (keyHandler){ window.removeEventListener('keydown', keyHandler, true); keyHandler = null; }
  activeId = null; pts = []; dragIdx = -1;
  requestStatic();
}
function clonePts(bend){
  const src = (bend && bend.length) ? bend : [{ beat: 0, semitones: 0 }];
  const out = src.map(p => ({ beat: p.beat, semitones: clamp(p.semitones, -MAXST, MAXST) }))
                 .sort((a, b) => a.beat - b.beat);
  out[0].beat = 0;   // pin the anchor
  return out;
}
function commit(){
  const n = note(); if (!n) return;
  apply(edit.setBend(n.id, pts));
}

// ---------- hit-testing ----------
function pickPoint(x, y){
  const n = note(); if (!n) return -1;
  let best = -1, bestD = HITR * HITR;
  for (let i = 0; i < pts.length; i++){
    const px = beatToXRel(n, pts[i].beat), py = semToY(n, pts[i].semitones);
    const d = (px - x) * (px - x) + (py - y) * (py - y);
    if (d <= bestD){ bestD = d; best = i; }
  }
  return best;
}
function inPanel(x, y){ const n = note(); if (!n) return false; const p = panel(n); return x >= p.x0 && x <= p.x1 && y >= p.y0 && y <= p.y1; }

// ---------- pointer (routed from interact.js) ----------
// returns true if the event was consumed by the editor
export function pointerDown(x, y, e){
  const n = note(); if (!n) return false;
  if (!inPanel(x, y)) return false;
  const idx = pickPoint(x, y);
  if (e.button === 2){                 // right-click removes a point (never the anchor)
    if (idx > 0){ pts.splice(idx, 1); commit(); }
    requestStatic();
    return true;
  }
  if (e.button !== 0) return true;
  dragIdx = idx;                       // -1 = clicked empty panel (consume, no-op)
  requestStatic();
  return true;
}
export function pointerMove(x, y){
  const n = note(); if (!n || dragIdx < 0) return;
  const p = pts[dragIdx];
  p.semitones = yToSem(n, y);
  if (dragIdx > 0){
    const lo = pts[dragIdx - 1].beat + MINGAP;
    const hi = (dragIdx < pts.length - 1) ? pts[dragIdx + 1].beat - MINGAP : n.len_beats;
    p.beat = clamp(xToBeatRel(n, x), lo, Math.max(lo, hi));
  }
  requestStatic();
}
export function pointerUp(){
  if (dragIdx >= 0){ commit(); dragIdx = -1; }
}
export function addPointAt(x, y){
  const n = note(); if (!n) return;
  if (!inPanel(x, y)) return;
  const bRel = xToBeatRel(n, x);
  const s = yToSem(n, y);
  // skip if essentially on an existing point's beat
  if (pts.some(p => Math.abs(p.beat - bRel) < 0.02)) return;
  pts.push({ beat: bRel, semitones: s });
  pts.sort((a, b) => a.beat - b.beat);
  pts[0].beat = 0;
  commit();
  requestStatic();
}

// ---------- draw (called from render.drawOverlay) ----------
export function draw(g){
  if (activeId == null) return;
  const n = note();
  if (!n){ close(); return; }
  if (dragIdx < 0) pts = clonePts(n.bend);   // stay in sync with external undo/redo

  const p = panel(n);
  const w = p.x1 - p.x0, h = p.y1 - p.y0;
  const cy = centerY(n), span = halfSpan();

  // backdrop panel
  g.fillStyle = 'rgba(12,16,20,0.72)'; g.fillRect(p.x0, p.y0, w, h);
  g.strokeStyle = 'rgba(255,255,255,0.14)'; g.lineWidth = 1; g.strokeRect(p.x0 + 0.5, p.y0 + 0.5, w - 1, h - 1);

  // gridlines: 0 (bright), ±1 (dim), ±2 (dashed edge)
  for (const s of [-2, -1, 0, 1, 2]){
    const y = Math.round(cy - (s / MAXST) * span) + 0.5;
    if (s === 0){ g.strokeStyle = 'rgba(255,255,255,0.30)'; g.setLineDash([]); }
    else if (Math.abs(s) === 2){ g.strokeStyle = 'rgba(255,255,255,0.16)'; g.setLineDash([2, 3]); }
    else { g.strokeStyle = 'rgba(255,255,255,0.08)'; g.setLineDash([]); }
    g.beginPath(); g.moveTo(p.x0, y); g.lineTo(p.x1, y); g.stroke();
    g.setLineDash([]);
    if (s !== 0){ g.fillStyle = 'rgba(200,200,200,0.5)'; g.font = '8px ui-monospace,monospace'; g.textBaseline = 'middle';
      g.fillText((s > 0 ? '+' : '') + s, p.x0 + 2, y); }
  }

  // curve polyline
  g.strokeStyle = '#ffd479'; g.lineWidth = 1.6; g.beginPath();
  let saturated = false;
  for (let i = 0; i < pts.length; i++){
    const px = beatToXRel(n, pts[i].beat), py = semToY(n, pts[i].semitones);
    if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
    if (Math.abs(pts[i].semitones) >= SAT) saturated = true;
  }
  // extend flat to the note end from the last point
  const last = pts[pts.length - 1];
  g.lineTo(beatToXRel(n, n.len_beats), semToY(n, last.semitones));
  g.stroke();

  // control points
  for (let i = 0; i < pts.length; i++){
    const px = beatToXRel(n, pts[i].beat), py = semToY(n, pts[i].semitones);
    g.beginPath(); g.arc(px, py, i === dragIdx ? 4.2 : 3.2, 0, Math.PI * 2);
    g.fillStyle = i === dragIdx ? '#fff' : '#ffcf6b'; g.fill();
    g.strokeStyle = 'rgba(0,0,0,0.7)'; g.lineWidth = 1; g.stroke();
  }

  // dragged-point value label
  if (dragIdx >= 0){
    const px = beatToXRel(n, pts[dragIdx].beat), py = semToY(n, pts[dragIdx].semitones);
    const s = pts[dragIdx].semitones;
    const txt = (s >= 0 ? '+' : '') + s.toFixed(2) + ' st';
    g.fillStyle = 'rgba(0,0,0,0.8)'; g.fillRect(px + 6, py - 14, 52, 13);
    g.fillStyle = '#ffd479'; g.font = '10px ui-monospace,monospace'; g.textBaseline = 'middle';
    g.fillText(txt, px + 9, py - 7);
  }

  // saturation warning glyph
  if (saturated){
    const wx = p.x1 - 12, wy = p.y0 + 8;
    g.fillStyle = '#e0a24f';
    g.beginPath(); g.moveTo(wx, wy - 5); g.lineTo(wx + 5, wy + 4); g.lineTo(wx - 5, wy + 4); g.closePath(); g.fill();
    g.fillStyle = '#1a1a1a'; g.font = 'bold 8px ui-sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('!', wx, wy + 1); g.textAlign = 'left';
  }
}
