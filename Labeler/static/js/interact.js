// interact.js — pointer + wheel handling on the stage: pan/zoom, click-select,
// playhead scrubbing, note drag-edit (move / re-pitch / resize),
// mute-region painting on the ruler, double-click create, and minimap navigation.
import {
  store, apply, select, setPlayhead, clamp, pitchName, noteById, notesSorted, nextNote,
} from './state.js';
import {
  layout, timeToX, xToTime, yToBin, binToY, noteRect, binForPitch, pitchForBin,
  visibleNotes, panBy, zoomAbout, clampView, durationS, binCount, ZOOM_STEP,
} from './timeline.js';
import * as edit from './editing.js';

let requestStatic = () => {};
let ptr = null;             // active pointer interaction
const EDGE_PX = 5, DRAG_PX = 3;

export function init(deps = {}){
  requestStatic = deps.requestStatic || (() => {});
  const roll = document.getElementById('roll');
  roll.addEventListener('wheel', onWheel, { passive: false });
  roll.addEventListener('pointerdown', onDown);
  roll.addEventListener('pointermove', onMove);
  roll.addEventListener('pointerup', onUp);
  roll.addEventListener('pointercancel', onUp);
  roll.addEventListener('dblclick', onDblClick);
  roll.addEventListener('pointerleave', () => { if (!ptr){ store.hover = null; } });

  const mm = document.getElementById('minimap');
  mm.addEventListener('pointerdown', onMiniDown);
  mm.addEventListener('pointermove', onMiniMove);
  mm.addEventListener('pointerup', () => { mmDrag = false; });
}

const editing = () => store.config && store.config.editing_enabled;

// ---------- geometry helpers ----------
function rollXY(e){
  const r = document.getElementById('roll').getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}
function bandOf(y){
  if (y < layout.rollTop) return 'ruler';
  if (y < layout.velTop) return 'roll';
  if (y < layout.onsetTop) return 'vel';
  return 'onset';
}

function hitNote(xInRoll, yInPiano){
  let body = null, edgeHit = null, edgeDist = Infinity;
  for (const n of visibleNotes()){
    const r = noteRect(n);
    if (yInPiano < r.y - 4 || yInPiano > r.y + r.h + 4) continue;
    if (editing()){
      // right-edge resize zone; clamp so short notes keep a grabbable left half
      const endX = r.x + r.w;
      const inner = Math.max(r.x + r.w / 2, endX - EDGE_PX);
      const d = Math.abs(xInRoll - endX);
      if (xInRoll >= inner && xInRoll <= endX + EDGE_PX && d <= edgeDist){
        edgeDist = d; edgeHit = { note: n, edge: 'end' };
      }
    }
    if (xInRoll < r.x - EDGE_PX || xInRoll > r.x + r.w + EDGE_PX) continue;
    body = { note: n, edge: null };     // last (topmost drawn) wins
  }
  return edgeHit || body;               // edge beats body: left note's end wins at a shared boundary
}
function noteAtTime(t){
  let best = null, bestw = Infinity;
  for (const n of visibleNotes()){
    if (t >= n.start_s && t <= n.end_s){ const w = n.end_s - n.start_s; if (w < bestw){ bestw = w; best = n; } }
  }
  return best;
}

// ---------- wheel: pan / zoom ----------
function onWheel(e){
  e.preventDefault();
  const scale = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? layout.width : 1;
  if (e.ctrlKey){
    const { x } = rollXY(e);
    const factor = (e.deltaY < 0) ? ZOOM_STEP : 1 / ZOOM_STEP;
    zoomAbout(x, factor);
  } else {
    const d = (e.deltaX !== 0 ? e.deltaX : e.deltaY) * scale;
    panBy(d / store.view.pxPerSec);
  }
  requestStatic();
}

// ---------- pointer ----------
function setCursor(name){ document.getElementById('roll').style.cursor = name || ''; }

function onDown(e){
  if (e.button !== 0) return;
  if (e.target.closest('.popover')) return;   // let popover buttons handle their own clicks
  document.getElementById('roll').setPointerCapture(e.pointerId);
  const { x, y } = rollXY(e);
  const band = bandOf(y);
  const t = clamp(xToTime(x), 0, durationS());

  if (band === 'ruler'){
    if (store.paintMode){ store.drag = { kind: 'region', start: t, cur: t }; ptr = { mode: 'region' }; requestStatic(); }
    else { setPlayhead(t); ptr = { mode: 'scrub' }; }
    return;
  }
  if (band === 'onset'){ setPlayhead(t); ptr = { mode: 'scrub' }; return; }

  if (band === 'vel'){
    // Envelope lane: click selects the note under the cursor (velocity is edited
    // via the inspector #insp-vel field). No drag-to-edit on this lane.
    const n = noteAtTime(t);
    if (n) select(n.id);
    ptr = { mode: 'idle' };
    return;
  }

  // roll band
  const yInPiano = y - layout.rollTop;
  const hit = hitNote(x, yInPiano);
  if (hit){
    select(hit.note.id, { playhead: hit.note.start_s });    // click note = select + playhead
    if (editing() && hit.edge){
      ptr = { mode: 'pending', drag: 'resize', id: hit.note.id, sx: e.clientX, sy: e.clientY };
    } else if (editing()){
      ptr = { mode: 'pending', drag: 'move', id: hit.note.id, sx: e.clientX, sy: e.clientY,
              orig: { start: hit.note.start_s, end: hit.note.end_s, pitch: hit.note.pitch } };
    } else ptr = { mode: 'click' };
  } else {
    setPlayhead(t);                                          // click empty = playhead only
    ptr = { mode: 'idle' };
  }
}

function onMove(e){
  const { x, y } = rollXY(e);
  const band = bandOf(y);
  // hover crosshair
  if (!ptr || ptr.mode === 'idle' || ptr.mode === 'click'){
    const t = xToTime(x);
    store.hover = { t, bin: band === 'roll' ? yToBin(y - layout.rollTop) : null };
    if (band === 'roll' && editing()){
      const hit = hitNote(x, y - layout.rollTop);
      setCursor(hit && hit.edge ? 'ew-resize' : (hit ? 'grab' : ''));
    } else setCursor(band === 'ruler' && store.paintMode ? 'crosshair' : '');
  }
  if (!ptr) return;

  if (ptr.mode === 'scrub'){ setPlayhead(clamp(xToTime(x), 0, durationS())); return; }
  if (ptr.mode === 'region'){ store.drag.cur = clamp(xToTime(x), 0, durationS()); return; }

  if (ptr.mode === 'pending'){
    if (Math.hypot(e.clientX - ptr.sx, e.clientY - ptr.sy) < DRAG_PX) return;
    // promote to a live drag
    ptr.mode = ptr.drag;
    store.drag = { kind: ptr.drag, id: ptr.id };
    requestStatic();          // hide the note on the notes layer (ghost takes over)
  }

  if (ptr.mode === 'move'){
    const n = ptr.orig;
    const dStart = (e.clientX - ptr.sx) / store.view.pxPerSec;
    const targetBin = yToBin(y - layout.rollTop);
    const targetPitch = clamp(pitchForBin(targetBin), 0, 127);
    const dPitch = targetPitch - n.pitch;
    ptr.pending = { dStart, dPitch };
    const r = ghostRect(n.start + dStart, n.end + dStart, n.pitch + dPitch);
    store.drag.previewRect = r;
    store.drag.pitchLabel = pitchName(n.pitch + dPitch);
    return;
  }
  if (ptr.mode === 'resize'){
    const dSec = (e.clientX - ptr.sx) / store.view.pxPerSec;
    ptr.pending = { dSec };
    const nn = noteById(ptr.id);
    const en = Math.max(nn.end_s + dSec, nn.start_s + 0.02);
    store.drag.previewRect = ghostRect(nn.start_s, en, nn.pitch);
    return;
  }
}

function ghostRect(startS, endS, pitch){
  const x = timeToX(startS);
  const w = Math.max(2, (endS - startS) * store.view.pxPerSec);
  const yc = binToY(binForPitch(pitch));
  const h = Math.max(3, 1.5 * (layout.rollH / (binCount() - 1)));
  return { x, y: yc - h / 2, w, h };
}

function onUp(e){
  if (!ptr){ return; }
  const p = ptr; ptr = null;
  try { document.getElementById('roll').releasePointerCapture(e.pointerId); } catch {}

  if (p.mode === 'move' && p.pending){
    const { dStart, dPitch } = p.pending;
    if (dStart !== 0 || dPitch !== 0) apply(edit.moveNote(p.id, dStart, dPitch, { phAfter: noteById(p.id).start_s + dStart }));
  } else if (p.mode === 'resize' && p.pending){
    if (p.pending.dSec !== 0) apply(edit.resizeNote(p.id, p.pending.dSec));
  } else if (p.mode === 'region' && store.drag){
    const a = store.drag.start, b = store.drag.cur;
    if (Math.abs(b - a) > 0.01) apply(edit.addMuteRegion({ start_s: a, end_s: b }));
  }
  store.drag = null;
  requestStatic();
}

function onDblClick(e){
  if (!editing()) return;
  const { x, y } = rollXY(e);
  if (bandOf(y) !== 'roll') return;
  if (hitNote(x, y - layout.rollTop)) return;      // only on empty space
  const t = clamp(xToTime(x), 0, durationS());
  const pitch = clamp(pitchForBin(yToBin(y - layout.rollTop)), 0, 127);
  // default 0.25s or until the next note starts
  let end = t + 0.25;
  for (const n of notesSorted()) if (n.start_s > t){ end = Math.min(end, n.start_s); break; }
  end = Math.max(end, t + 0.05);
  const cfg = store.config;
  const defTech = (cfg && cfg.default_articulation) || (cfg && cfg.articulations[0] && cfg.articulations[0].name) || 'normal';
  apply(edit.addNote({ start_s: t, end_s: end, pitch, technique: defTech, velocity: 80 }));
  requestStatic();
}

// ---------- minimap ----------
let mmDrag = false;
function jumpTo(e){
  const r = document.getElementById('minimap').getBoundingClientRect();
  const t = ((e.clientX - r.left) / r.width) * durationS();
  const viewW = layout.width / store.view.pxPerSec;
  store.view.originS = t - viewW / 2;
  clampView();
  requestStatic();
}
function onMiniDown(e){ mmDrag = true; jumpTo(e); }
function onMiniMove(e){ if (mmDrag) jumpTo(e); }
