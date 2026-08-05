// interact.js — pointer + wheel handling for the grid, piano-keys column, and velocity
// lane. Implements the tool state machine (draw/select/delete/slice), marquee, move,
// edge-resize, wheel pan, ctrl+wheel zoom-about-cursor. Modifier gestures: ctrl+drag =
// marquee multi-select (shift+ctrl adds to it); alt = temporary free snap for any drag,
// and alt+drag on a note body = free end-resize. Ctrl wins when both are held.
// Every mutation goes through editing.js commands via state.apply().
import {
  store, apply, clamp, noteById, notesSorted, selectedIds, isSelected,
  selectOnly, setSelection, toggleSelection, clearSelection, pitchName, notify,
} from './state.js';
import {
  xToBeat, beatToX, pitchToY, pitchRowAt, noteRect, rectFor, layout,
  visibleNotes, panByBeats, panByPitch, zoomAbout, clampView, ZOOM_STEP,
  PITCH_LO, PITCH_HI,
} from './timeline.js';
import { snapBeat, defaultLen } from './snap.js';
import * as edit from './editing.js';
import * as bendedit from './bendedit.js';
import { player } from './player.js';

let requestStatic = () => {};
let ptr = null;
const EDGE_PX = 6, DRAG_PX = 3;

export function init(deps = {}){
  requestStatic = deps.requestStatic || (() => {});
  const grid = document.getElementById('grid');
  grid.addEventListener('wheel', onWheel, { passive: false });
  grid.addEventListener('pointerdown', onDown);
  grid.addEventListener('pointermove', onMove);
  grid.addEventListener('pointerup', onUp);
  grid.addEventListener('pointercancel', onUp);
  grid.addEventListener('contextmenu', (e) => e.preventDefault());
  grid.addEventListener('pointerleave', () => { if (!ptr){ store.hover = null; } });
  grid.addEventListener('dblclick', onDblClick);

  const keys = document.getElementById('keys');
  keys.addEventListener('pointerdown', onKeyDown);
  // the control lane (velocity/pan/pitch) owns its own pointer handling in vellane.js
}

// ---------- double-click: open the per-note pitch-bend editor ----------
function onDblClick(e){
  if (e.button !== 0) return;
  const grid = document.getElementById('grid');
  const { x, y } = xy(grid, e);
  if (bendedit.isOpen()){ bendedit.addPointAt(x, y); return; }
  const hit = hitNote(x, y);
  if (hit){ selectOnly(hit.note.id); bendedit.open(hit.note, { requestStatic }); }
}

// ---------- geometry ----------
function xy(el, e){ const r = el.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

function hitNote(x, y){
  let best = null;
  for (const n of visibleNotes()){
    const r = noteRect(n);
    if (y < r.y || y > r.y + r.h) continue;
    if (x < r.x - EDGE_PX || x > r.x + r.w + EDGE_PX) continue;
    let edge = null;
    if (Math.abs(x - r.x) <= EDGE_PX) edge = 'start';
    else if (Math.abs(x - (r.x + r.w)) <= EDGE_PX) edge = 'end';
    best = { note: n, edge };   // topmost drawn wins
  }
  return best;
}
// ---------- wheel ----------
function onWheel(e){
  e.preventDefault();
  const grid = document.getElementById('grid');
  if (e.ctrlKey){
    const { x } = xy(grid, e);
    zoomAbout(x, e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
  } else if (e.shiftKey){
    panByPitch(e.deltaY < 0 ? 2 : -2);
  } else {
    const d = (e.deltaX !== 0 ? e.deltaX : e.deltaY);
    panByBeats(d / store.view.px_per_beat);
  }
  requestStatic();
}

// ---------- pointer down ----------
function onDown(e){
  const grid = document.getElementById('grid');
  const { x, y } = xy(grid, e);
  if (y < 0 || y >= layout.gridH || x < 0) return;
  const beat = xToBeat(x), pitch = pitchRowAt(y);
  const right = e.button === 2;
  grid.setPointerCapture(e.pointerId);

  // bend editor (when open) intercepts pointer input over its panel; a click outside
  // commits and closes it, consuming the event.
  if (bendedit.isOpen()){
    if (bendedit.pointerDown(x, y, e)){ ptr = { mode: 'bend' }; return; }
    bendedit.close(); ptr = { mode: 'idle' }; requestStatic(); return;
  }

  // right-click anywhere = delete the note under the cursor
  if (right){
    const hit = hitNote(x, y);
    if (hit) apply(edit.deleteNotes([hit.note.id]));
    ptr = { mode: 'idle' };
    return;
  }
  if (e.button !== 0) return;

  const hit = hitNote(x, y);
  const tool = store.tool;

  // ctrl+drag marquee multi-select (works over notes); shift+ctrl adds to selection
  if (e.ctrlKey && (tool === 'draw' || tool === 'select')){
    if (!e.shiftKey) clearSelection();
    ptr = { mode: 'marquee', base: new Set(selectedIds()) };
    store.drag = { kind: 'marquee', x0: x, y0: y, x1: x, y1: y };
    requestStatic();
    return;
  }
  // alt+drag on a note body = free end-resize (ctrl wins if both held)
  if (e.altKey && hit && (tool === 'draw' || tool === 'select')){
    if (!isSelected(hit.note.id)) selectOnly(hit.note.id);
    store.altFree = true;
    startResize(hit.note, hit.edge || 'end', e);
    return;
  }

  if (tool === 'delete'){
    if (hit) apply(edit.deleteNotes([hit.note.id]));
    ptr = { mode: 'delete-paint' };
    return;
  }
  if (tool === 'slice'){
    if (hit){ const c = edit.sliceNote(hit.note.id, snapBeat(beat)); if (c) apply(c); }
    ptr = { mode: 'idle' };
    return;
  }

  if (tool === 'draw'){
    if (hit && hit.edge){ if (!isSelected(hit.note.id)) selectOnly(hit.note.id); startResize(hit.note, hit.edge, e); return; }
    if (hit){ if (!isSelected(hit.note.id)) selectOnly(hit.note.id); startMove(hit.note, e, false); return; }
    // create a new note and immediately drag its end edge to set length
    const start = Math.max(0, snapBeat(beat));
    const len = defaultLen();
    const note = edit.makeNote({ start_beat: start, len_beats: len, pitch: clamp(pitch, PITCH_LO, PITCH_HI), velocity: 100 });
    apply(edit.createNotes([note]));
    startResize(note, 'end', e, { fresh: true });
    return;
  }

  // select tool
  if (hit){
    if (e.shiftKey){ toggleSelection(hit.note.id); ptr = { mode: 'idle' }; return; }
    if (!isSelected(hit.note.id)) selectOnly(hit.note.id);
    startMove(hit.note, e, false);
    return;
  }
  // marquee
  if (!e.shiftKey) clearSelection();
  ptr = { mode: 'marquee', base: new Set(selectedIds()) };
  store.drag = { kind: 'marquee', x0: x, y0: y, x1: x, y1: y };
  requestStatic();
}

function startMove(note, e, fresh){
  const grid = document.getElementById('grid');
  const { x, y } = xy(grid, e);
  const ids = isSelected(note.id) ? selectedIds() : [note.id];
  ptr = {
    mode: 'pending', promote: 'move', ids, anchorId: note.id,
    sx: e.clientX, sy: e.clientY, startBeat: xToBeat(x), startPitch: pitchRowAt(y),
    orig: ids.map(id => { const n = noteById(id); return { id, start: n.start_beat, pitch: n.pitch, len: n.len_beats }; }),
  };
}
function startResize(note, edge, e, opts = {}){
  const ids = isSelected(note.id) ? selectedIds() : [note.id];
  ptr = {
    mode: opts.fresh ? 'resize' : 'pending', promote: 'resize', ids, edge, anchorId: note.id,
    sx: e.clientX, fresh: !!opts.fresh,
    orig: ids.map(id => { const n = noteById(id); return { id, start: n.start_beat, len: n.len_beats }; }),
  };
  if (opts.fresh){ store.drag = { kind: 'resize', hideIds: new Set(ids), ghosts: [] }; requestStatic(); }
}

// ---------- pointer move ----------
function onMove(e){
  const grid = document.getElementById('grid');
  const { x, y } = xy(grid, e);

  if (ptr && ptr.mode === 'bend'){ bendedit.pointerMove(x, y); return; }

  // alt held/released mid-drag toggles free snap live (move/resize/create)
  if (ptr && ptr.mode !== 'idle'){ store.altFree = e.altKey; }

  if (!ptr || ptr.mode === 'idle'){
    store.hover = { beat: xToBeat(x), pitch: pitchRowAt(y) };
    if (store.tool === 'select' || store.tool === 'draw'){
      const hit = hitNote(x, y);
      grid.style.cursor = hit && hit.edge ? 'ew-resize' : (hit ? 'move' : (store.tool === 'draw' ? 'crosshair' : 'default'));
    }
    return;
  }

  if (ptr.mode === 'marquee'){
    store.drag.x1 = x; store.drag.y1 = y;
    updateMarquee();
    return;
  }
  if (ptr.mode === 'delete-paint'){
    const hit = hitNote(x, y);
    if (hit) apply(edit.deleteNotes([hit.note.id]));
    return;
  }
  if (ptr.mode === 'pending'){
    if (Math.hypot(e.clientX - ptr.sx, e.clientY - (ptr.sy || e.clientY)) < DRAG_PX) return;
    ptr.mode = ptr.promote;
    store.drag = { kind: ptr.promote, hideIds: new Set(ptr.ids), ghosts: [] };
    requestStatic();
  }

  if (ptr.mode === 'move'){
    const rawStart = ptr.orig.find(o => o.id === ptr.anchorId).start + (xToBeat(x) - ptr.startBeat);
    const anchorSnapped = Math.max(0, snapBeat(rawStart));
    const dBeat = anchorSnapped - ptr.orig.find(o => o.id === ptr.anchorId).start;
    const dPitch = pitchRowAt(y) - ptr.startPitch;
    ptr.dBeat = dBeat; ptr.dPitch = dPitch;
    store.drag.ghosts = ptr.orig.map(o => rectFor(Math.max(0, o.start + dBeat), o.len, clamp(o.pitch + dPitch, 0, 127)));
    store.drag.label = pitchName(clamp(ptr.orig.find(o => o.id === ptr.anchorId).pitch + dPitch, 0, 127));
    return;
  }
  if (ptr.mode === 'resize'){
    const anchor = ptr.orig.find(o => o.id === ptr.anchorId);
    const cursorBeat = xToBeat(x);
    let dBeat;
    if (ptr.edge === 'end'){ dBeat = Math.max(0, snapBeat(cursorBeat)) - (anchor.start + anchor.len); }
    else { dBeat = Math.max(0, snapBeat(cursorBeat)) - anchor.start; }
    ptr.dBeat = dBeat;
    store.drag.ghosts = ptr.orig.map(o => {
      if (ptr.edge === 'end') return rectFor(o.start, Math.max(0.03125, o.len + dBeat), noteById(o.id).pitch);
      const ns = clamp(o.start + dBeat, 0, o.start + o.len - 0.03125);
      return rectFor(ns, (o.start + o.len) - ns, noteById(o.id).pitch);
    });
    store.drag.label = null;
    return;
  }
}

function updateMarquee(){
  const d = store.drag;
  const bx0 = Math.min(d.x0, d.x1), bx1 = Math.max(d.x0, d.x1);
  const by0 = Math.min(d.y0, d.y1), by1 = Math.max(d.y0, d.y1);
  const inside = [];
  for (const n of visibleNotes()){
    const r = noteRect(n);
    if (r.x + r.w >= bx0 && r.x <= bx1 && r.y + r.h >= by0 && r.y <= by1) inside.push(n.id);
  }
  setSelection([...ptr.base, ...inside]);
}

// ---------- pointer up ----------
function onUp(e){
  if (!ptr){ return; }
  const p = ptr; ptr = null;
  try { document.getElementById('grid').releasePointerCapture(e.pointerId); } catch {}
  store.altFree = false;

  if (p.mode === 'bend'){ bendedit.pointerUp(); requestStatic(); return; }

  if (p.mode === 'move' && (p.dBeat || p.dPitch)){
    const anchor = p.orig.find(o => o.id === p.anchorId);
    apply(edit.moveNotes(p.ids, p.dBeat || 0, p.dPitch || 0, { phAfter: Math.max(0, anchor.start + (p.dBeat || 0)) }));
  } else if (p.mode === 'resize'){
    const anchor = p.orig.find(o => o.id === p.anchorId);
    if (p.dBeat){ apply(edit.resizeNotes(p.ids, p.edge, p.dBeat)); }
    // remember the last-used length from the freshly drawn / resized anchor
    const n = noteById(p.anchorId); if (n) store.lastLen = n.len_beats;
  }
  store.drag = null;
  requestStatic();
  notify();
}

// ---------- piano keys: click = preview ----------
function onKeyDown(e){
  if (e.button !== 0) return;
  const keys = document.getElementById('keys');
  const { y } = xy(keys, e);
  const pitch = pitchRowAt(y);
  try { player.previewNote(pitch); } catch {}
}
