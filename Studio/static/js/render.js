// render.js — all canvas drawing, DPR-correct. Stacked canvases:
//   #ruler #keys #grid #lane #wave #overlay #minimap
// #ruler/#keys/#grid/#lane/#minimap redraw on view/data changes (drawStage);
// #overlay redraws every rAF (drawOverlay). #wave is a Phase-4 placeholder.
import { store, pitchName, isBlackKey, clamp, isSelected } from './state.js';
import {
  layout, beatToX, xToBeat, pitchToY, noteRect, visibleNotes,
  visibleBeatRange, visiblePitchRange, durationBeats, clampView, PITCH_LO, PITCH_HI,
} from './timeline.js';
import { gridSize } from './snap.js';
import * as vellane from './vellane.js';
import * as bendedit from './bendedit.js';
import * as waveform from './waveform.js';

const els = {}, cx = {};
let onRedraw = () => {};

// ---------- FL charcoal palette ----------
const COL = {
  gridWhite: '#35454f',   // white-key row
  gridBlack: '#2c3a43',   // black-key row (darker)
  lineBar:  'rgba(0,0,0,0.55)',
  lineBeat: 'rgba(0,0,0,0.34)',
  lineStep: 'rgba(255,255,255,0.055)',
  rulerBg:  '#232323',
  keyWhite: '#d9d9d9',
  keyBlack: '#1b1b1b',
  keyFace:  '#cfcfcf',
  laneBg:   '#1b2228',
  waveBg:   '#181818',
  mmBg:     '#202020',
  noteDefault: '#5ac54f',
  noteSelect:  '#e0574d',
  playhead: '#e8e8e8',
};

export function init(redrawCb){
  onRedraw = redrawCb || (() => {});
  for (const id of ['ruler', 'keys', 'grid', 'lane', 'wave', 'overlay', 'minimap']){
    const el = document.getElementById(id);
    els[id] = el; cx[id] = el.getContext('2d');
  }
  wireMinimap();
}

// Minimap click / drag: center the main view on the clicked beat (FL-style jump-scroll).
function wireMinimap(){
  const mm = els.minimap; if (!mm) return;
  let dragging = false;
  const jump = (e) => {
    const r = mm.getBoundingClientRect();
    const frac = clamp((e.clientX - r.left) / Math.max(1, r.width), 0, 1);
    const beat = frac * (durationBeats() || 1);
    const viewBeats = layout.gridW / store.view.px_per_beat;
    store.view.scroll_beat = beat - viewBeats / 2;
    clampView(); onRedraw();
  };
  mm.addEventListener('pointerdown', (e) => { if (e.button !== 0) return; dragging = true; mm.setPointerCapture(e.pointerId); jump(e); });
  mm.addEventListener('pointermove', (e) => { if (dragging) jump(e); });
  mm.addEventListener('pointerup', (e) => { dragging = false; try { mm.releasePointerCapture(e.pointerId); } catch {} });
}

// ---------- sizing ----------
export function resize(){
  const dpr = window.devicePixelRatio || 1;
  layout.dpr = dpr;
  function size(id){
    const el = els[id]; if (!el) return { width: 1, height: 1 };
    const r = el.getBoundingClientRect();
    el.width = Math.max(1, Math.round(r.width * dpr));
    el.height = Math.max(1, Math.round(r.height * dpr));
    cx[id].setTransform(dpr, 0, 0, dpr, 0, 0);
    return r;
  }
  size('ruler');
  const rKeys = size('keys');
  const rGrid = size('grid');
  size('lane'); size('wave'); size('overlay');
  const rMm = size('minimap');
  layout.keyW = rKeys.width;
  layout.rulerH = els.ruler.getBoundingClientRect().height;
  layout.gridW = rGrid.width; layout.gridH = rGrid.height;
  layout.laneH = els.lane.getBoundingClientRect().height;
  layout.waveH = els.wave.getBoundingClientRect().height;
  layout.mmW = rMm.width; layout.mmH = rMm.height;
}

const crisp = (v) => Math.round(v) + 0.5;

// ---------- technique color palette (Phase 2 plugs in via config.articulations) ----------
let palette = null;
export function techColor(name){
  if (!palette){
    palette = new Map();
    for (const a of (store.config && store.config.articulations) || []) palette.set(a.name, a.color);
  }
  return palette.get(name) || COL.noteDefault;
}
export function resetColors(){ palette = null; }

// hex shade helper (amt>0 lighten toward white, <0 darken toward black)
export function shade(hex, amt){
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  const t = amt < 0 ? 0 : 255, p = Math.abs(amt);
  r = Math.round(r + (t - r) * p); g = Math.round(g + (t - g) * p); b = Math.round(b + (t - b) * p);
  return `rgb(${r},${g},${b})`;
}

// blend two #rrggbb hex colors, t in [0,1] toward `b`. Returns rgb() string.
export function mix(a, b, t){
  const pa = /^#?([0-9a-f]{6})$/i.exec(a), pb = /^#?([0-9a-f]{6})$/i.exec(b);
  if (!pa || !pb) return a;
  const na = parseInt(pa[1], 16), nb = parseInt(pb[1], 16);
  const r = Math.round(((na >> 16) & 255) * (1 - t) + ((nb >> 16) & 255) * t);
  const g = Math.round(((na >> 8) & 255) * (1 - t) + ((nb >> 8) & 255) * t);
  const bl = Math.round((na & 255) * (1 - t) + (nb & 255) * t);
  return `rgb(${r},${g},${bl})`;
}
export const SELECT_COL = COL.noteSelect;

// ---------- public draw entry ----------
export function drawStage(){
  drawRuler();
  drawKeys();
  drawGrid();
  drawLane();
  drawWave();
  drawMinimap();
}

// ---------- ruler ----------
function drawRuler(){
  const g = cx.ruler, w = layout.gridW, h = layout.rulerH;
  g.clearRect(0, 0, w, h);
  g.fillStyle = COL.rulerBg; g.fillRect(0, 0, w, h);
  const [b0, b1] = visibleBeatRange();
  const num = (store.doc && store.doc.time_sig && store.doc.time_sig.num) || 4;
  const pxb = store.view.px_per_beat;

  // loop region highlight (top half)
  if (store.loop.enabled){
    const lx = beatToX(store.loop.start_beat), lx2 = beatToX(store.loop.end_beat);
    g.fillStyle = 'rgba(90,197,79,0.22)'; g.fillRect(lx, 0, lx2 - lx, h * 0.5);
    g.strokeStyle = 'rgba(90,197,79,0.85)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(crisp(lx), 0); g.lineTo(crisp(lx), h * 0.5);
    g.moveTo(crisp(lx2), 0); g.lineTo(crisp(lx2), h * 0.5); g.stroke();
  }

  // adaptive beat ticks (skip labels when too dense)
  const labelEveryBeat = pxb >= 28;
  g.font = '10px ui-monospace,monospace'; g.textBaseline = 'middle';
  const firstBeat = Math.floor(b0);
  for (let beat = firstBeat; beat <= b1 + 1; beat++){
    const x = crisp(beatToX(beat));
    const isBar = beat % num === 0;
    g.strokeStyle = isBar ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.14)';
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, isBar ? h * 0.4 : h * 0.62); g.lineTo(x, h); g.stroke();
    if (isBar){
      const bar = Math.floor(beat / num) + 1;
      g.fillStyle = '#c8c8c8'; g.fillText(String(bar), x + 3, h * 0.5);
    } else if (labelEveryBeat){
      g.fillStyle = '#7a7a7a'; g.fillText(String((beat % num) + 1), x + 3, h * 0.62);
    }
  }
  // playhead marker
  const px = crisp(beatToX(store.playhead));
  g.fillStyle = COL.playhead;
  g.beginPath(); g.moveTo(px - 4, 0); g.lineTo(px + 4, 0); g.lineTo(px, 7); g.fill();
}

// ---------- piano keyboard column ----------
function drawKeys(){
  const g = cx.keys, w = layout.keyW, h = layout.gridH;
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#161616'; g.fillRect(0, 0, w, h);
  const [pLo, pHi] = visiblePitchRange();
  g.font = '9px ui-sans-serif,system-ui'; g.textBaseline = 'middle';
  for (let p = pHi + 1; p >= pLo - 1; p--){
    if (p < PITCH_LO || p > PITCH_HI) continue;
    const y = pitchToY(p);
    const black = isBlackKey(p);
    g.fillStyle = black ? COL.keyBlack : COL.keyWhite;
    g.fillRect(0, y, w, layout.rowH);
    g.strokeStyle = 'rgba(0,0,0,0.4)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, crisp(y)); g.lineTo(w, crisp(y)); g.stroke();
    if (p % 12 === 0){   // label C's
      g.fillStyle = '#333'; g.fillText(pitchName(p), 4, y + layout.rowH / 2);
    }
  }
  g.strokeStyle = '#000'; g.beginPath(); g.moveTo(crisp(w), 0); g.lineTo(crisp(w), h); g.stroke();
}

// ---------- grid + notes ----------
function drawGrid(){
  const g = cx.grid, w = layout.gridW, h = layout.gridH;
  g.clearRect(0, 0, w, h);
  const [pLo, pHi] = visiblePitchRange();

  // alternating black/white-key row shading
  for (let p = pHi + 1; p >= pLo - 1; p--){
    const y = pitchToY(p);
    g.fillStyle = isBlackKey(p) ? COL.gridBlack : COL.gridWhite;
    g.fillRect(0, y, w, layout.rowH);
  }
  // horizontal row separators + octave (C) emphasis
  for (let p = pHi + 1; p >= pLo - 1; p--){
    const y = crisp(pitchToY(p));
    g.strokeStyle = p % 12 === 0 ? 'rgba(0,0,0,0.4)' : 'rgba(0,0,0,0.16)';
    g.lineWidth = 1; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }

  // three-tier vertical lines: bar / beat / step
  const [b0, b1] = visibleBeatRange();
  const num = (store.doc && store.doc.time_sig && store.doc.time_sig.num) || 4;
  const step = gridSize('step') || 0.25;
  const pxStep = step * store.view.px_per_beat;
  // step lines only when they won't smear
  if (pxStep >= 5){
    g.strokeStyle = COL.lineStep; g.lineWidth = 1;
    const first = Math.ceil(b0 / step) * step;
    g.beginPath();
    for (let bt = first; bt <= b1; bt += step){
      if (Math.abs(bt % 1) < 1e-6) continue;   // beat/bar drawn below
      const x = crisp(beatToX(bt)); g.moveTo(x, 0); g.lineTo(x, h);
    }
    g.stroke();
  }
  g.strokeStyle = COL.lineBeat; g.lineWidth = 1; g.beginPath();
  for (let beat = Math.ceil(b0); beat <= b1; beat++){
    if (beat % num === 0) continue;
    const x = crisp(beatToX(beat)); g.moveTo(x, 0); g.lineTo(x, h);
  }
  g.stroke();
  g.strokeStyle = COL.lineBar; g.lineWidth = 1; g.beginPath();
  const firstBar = Math.ceil(b0 / num) * num;
  for (let beat = firstBar; beat <= b1; beat += num){
    const x = crisp(beatToX(beat)); g.moveTo(x, 0); g.lineTo(x, h);
  }
  g.stroke();

  // notes (skip ones being dragged — the overlay draws their ghost)
  const dragIds = store.drag && store.drag.hideIds ? store.drag.hideIds : null;
  for (const n of visibleNotes()){
    if (dragIds && dragIds.has(n.id)) continue;
    const r = noteRect(n);
    drawNote(g, r, techColor(n.technique), isSelected(n.id));
    drawNoteGlyphs(g, r, n);
  }
}

// FL-style flat-beveled note: dark outline, flat fill, lighter top edge, darker bottom.
// Selection is a red tint *blended over* the technique fill (keeps the articulation hue
// readable) plus a white outline, FL-style.
export function drawNote(g, r, base, selected){
  const col = selected ? mix(base, COL.noteSelect, 0.6) : base;
  const x = Math.round(r.x), y = Math.round(r.y), w = Math.max(2, Math.round(r.w)), h = Math.round(r.h);
  g.fillStyle = shade(col, -0.55); g.fillRect(x, y, w, h);           // outline
  g.fillStyle = col; g.fillRect(x + 1, y + 1, w - 2, h - 2);          // face
  g.fillStyle = shade(col, 0.35); g.fillRect(x + 1, y + 1, w - 2, Math.max(1, (h - 2) * 0.28)); // top light
  g.fillStyle = shade(col, -0.28); g.fillRect(x + 1, y + h - 2, w - 2, 1);                        // bottom dark
  if (selected){ g.strokeStyle = '#fff'; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
}

// small per-note conditioning indicators: a wavy "V" when vibrato depth>0, and a "~"
// bend marker when the note carries a non-trivial pitch-bend curve.
function drawNoteGlyphs(g, r, n){
  const x = Math.round(r.x), y = Math.round(r.y), w = Math.max(2, Math.round(r.w)), h = Math.round(r.h);
  if (w < 10 || h < 7) return;
  const vib = n.vibrato && n.vibrato.depth_semitones > 0;
  const bendPts = n.bend || [];
  const hasBend = bendPts.length > 1 || (bendPts.length === 1 && Math.abs(bendPts[0].semitones) > 1e-3);
  if (vib){
    // little wavy glyph near the note's right edge, vertically centered
    const gy = y + h / 2, gx = x + w - 9;
    g.strokeStyle = 'rgba(255,255,255,0.72)'; g.lineWidth = 1;
    g.beginPath();
    g.moveTo(gx, gy + 1.6);
    g.quadraticCurveTo(gx + 1.6, gy - 2.4, gx + 3.2, gy);
    g.quadraticCurveTo(gx + 4.8, gy + 2.4, gx + 6.4, gy - 1.6);
    g.stroke();
  }
  if (hasBend){
    const gy = y + 2.5, gx = x + 3;
    g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = 1;
    g.beginPath();
    g.moveTo(gx, gy + 1.4);
    g.quadraticCurveTo(gx + 2, gy - 1.6, gx + 4, gy + 0.2);
    g.quadraticCurveTo(gx + 5.4, gy + 1.6, gx + 7, gy - 1.2);
    g.stroke();
  }
}

// ---------- control lane (velocity / pan / pitch) — owned by vellane.js ----------
function drawLane(){
  vellane.draw(cx.lane, layout.gridW, layout.laneH);
}

// ---------- waveform strip (owned by waveform.js) ----------
function drawWave(){
  waveform.draw(cx.wave, layout.gridW, layout.waveH);
}

// ---------- minimap ----------
function drawMinimap(){
  const g = cx.minimap, w = layout.mmW, h = layout.mmH;
  g.clearRect(0, 0, w, h);
  g.fillStyle = COL.mmBg; g.fillRect(0, 0, w, h);
  const dur = durationBeats() || 1;
  const ns = (store.doc && store.doc.notes) || [];
  let pmin = PITCH_HI, pmax = PITCH_LO;
  for (const n of ns){ pmin = Math.min(pmin, n.pitch); pmax = Math.max(pmax, n.pitch); }
  if (pmax < pmin){ pmin = 55; pmax = 88; }
  const span = Math.max(1, pmax - pmin);
  for (const n of ns){
    const x = (n.start_beat / dur) * w;
    const x2 = ((n.start_beat + n.len_beats) / dur) * w;
    const y = h - 3 - ((n.pitch - pmin) / span) * (h - 7);
    g.strokeStyle = isSelected(n.id) ? COL.noteSelect : techColor(n.technique);
    g.globalAlpha = 0.9; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(x, y); g.lineTo(Math.max(x + 1, x2), y); g.stroke();
  }
  g.globalAlpha = 1;
  const [b0, b1] = visibleBeatRange();
  const vx = (b0 / dur) * w, vw = ((b1 - b0) / dur) * w;
  g.fillStyle = 'rgba(90,197,79,0.12)'; g.fillRect(vx, 0, vw, h);
  g.strokeStyle = 'rgba(90,197,79,0.7)'; g.lineWidth = 1; g.strokeRect(crisp(vx), 0.5, vw, h - 1);
  const px = (store.playhead / dur) * w;
  g.strokeStyle = COL.playhead; g.beginPath(); g.moveTo(crisp(px), 0); g.lineTo(crisp(px), h); g.stroke();
}

// ---------- overlay (rAF) ----------
export function drawOverlay(){
  const g = cx.overlay, w = layout.gridW, h = layout.gridH;
  g.clearRect(0, 0, w, h);

  // hover crosshair
  if (store.hover && store.hover.pitch != null && (!store.drag)){
    const y = crisp(pitchToY(store.hover.pitch));
    g.strokeStyle = 'rgba(255,255,255,0.10)'; g.lineWidth = 1;
    g.fillStyle = 'rgba(255,255,255,0.04)'; g.fillRect(0, pitchToY(store.hover.pitch), w, layout.rowH);
  }

  // drag ghosts
  if (store.drag) drawDrag(g);

  // per-note bend polyline editor (when open)
  bendedit.draw(g);

  // playhead
  const px = crisp(beatToX(store.playhead));
  g.strokeStyle = COL.playhead; g.lineWidth = 1;
  g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
}

function drawDrag(g){
  const d = store.drag;
  if (d.kind === 'marquee'){
    const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
    const rw = Math.abs(d.x1 - d.x0), rh = Math.abs(d.y1 - d.y0);
    g.fillStyle = 'rgba(224,87,77,0.14)'; g.fillRect(x, y, rw, rh);
    g.strokeStyle = 'rgba(224,87,77,0.8)'; g.lineWidth = 1; g.strokeRect(crisp(x), crisp(y), rw, rh);
    return;
  }
  if (d.ghosts){
    for (const r of d.ghosts){
      g.strokeStyle = '#fff'; g.lineWidth = 1.4; g.setLineDash([4, 3]);
      g.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h); g.setLineDash([]);
    }
    if (d.label){
      const rx = d.ghosts[0].x, ry = d.ghosts[0].y;
      g.fillStyle = 'rgba(0,0,0,0.8)'; g.fillRect(rx, ry - 15, 34, 13);
      g.fillStyle = '#8ee08a'; g.font = '10px ui-monospace,monospace'; g.textBaseline = 'middle';
      g.fillText(d.label, rx + 3, ry - 8);
    }
  }
}
