// timeline.js — beat<->pixel and pitch<->pixel transforms, shared layout geometry,
// pan/zoom clamps, and binary-search note culling. Grid is beats on X and linear
// semitone rows on Y (constant row height, high pitch at top).
import { store, clamp, notesSorted, noteEnd, timeSig } from './state.js';

export const PX_MIN = 12, PX_MAX = 400, ZOOM_STEP = 1.2;
export const PITCH_LO = 33, PITCH_HI = 100;   // scrollable pitch domain (violin range + margin)

// Live pixel geometry, filled by render.resize().
export const layout = {
  dpr: 1,
  keyW: 62,
  rulerH: 24,
  gridW: 800, gridH: 400,
  rowH: 13,
  laneH: 110,
  waveH: 46,
  mmW: 800, mmH: 26,
};

export function visibleRows(){ return Math.max(1, layout.gridH / layout.rowH); }

// ---------- beat <-> x ----------
export const beatToX = (beat) => (beat - store.view.scroll_beat) * store.view.px_per_beat;
export const xToBeat = (x) => store.view.scroll_beat + x / store.view.px_per_beat;
export function visibleBeatRange(){
  const b0 = store.view.scroll_beat;
  return [b0, b0 + layout.gridW / store.view.px_per_beat];
}

// ---------- pitch <-> y (high pitch at top) ----------
export const pitchToY = (pitch) => (store.view.top_pitch - pitch) * layout.rowH;
export const yToPitchF = (y) => store.view.top_pitch - y / layout.rowH;             // fractional
export const yToPitch = (y) => Math.round(store.view.top_pitch - y / layout.rowH - 0.5) + 0; // row -> integer pitch
export function pitchRowAt(y){ return store.view.top_pitch - Math.floor(y / layout.rowH); }
export function visiblePitchRange(){
  const top = store.view.top_pitch;
  const bottom = Math.floor(top - visibleRows());
  return [bottom, top];
}

// ---------- note rectangle (shared by render + hit-testing) ----------
export function noteRect(note){
  const x = beatToX(note.start_beat);
  const w = Math.max(2, note.len_beats * store.view.px_per_beat);
  const y = pitchToY(note.pitch);
  return { x, y, w, h: layout.rowH };
}
// arbitrary (start,len,pitch) -> rect (for drag ghosts / previews)
export function rectFor(startBeat, lenBeats, pitch){
  const x = beatToX(startBeat);
  const w = Math.max(2, lenBeats * store.view.px_per_beat);
  return { x, y: pitchToY(pitch), w, h: layout.rowH };
}

// velocity-lane bar geometry (bar + circle handle)
export function velBar(note){
  const x = beatToX(note.start_beat);
  const w = Math.max(2, note.len_beats * store.view.px_per_beat);
  const frac = clamp(note.velocity / 127, 0, 1);
  const h = frac * (layout.laneH - 8);
  return { x, w, top: layout.laneH - h, frac };
}

// ---------- content extent ----------
export function durationBeats(){
  let end = timeSig().num * 4;   // at least a few bars
  for (const n of notesSorted()) end = Math.max(end, noteEnd(n));
  return end + timeSig().num;    // one trailing bar of headroom
}

// ---------- pan / zoom (not undoable) ----------
export function clampView(){
  const v = store.view;
  v.px_per_beat = clamp(v.px_per_beat, PX_MIN, PX_MAX);
  const viewBeats = layout.gridW / v.px_per_beat;
  const maxScroll = Math.max(0, durationBeats() - viewBeats * 0.5);
  v.scroll_beat = clamp(v.scroll_beat, 0, maxScroll);
  const rows = visibleRows();
  const topMin = Math.min(PITCH_HI, PITCH_LO + rows);
  v.top_pitch = clamp(v.top_pitch, topMin, PITCH_HI);
}
export function panByBeats(dBeat){ store.view.scroll_beat += dBeat; clampView(); }
export function panByPitch(dPitch){ store.view.top_pitch += dPitch; clampView(); }
export function zoomAbout(cursorX, factor){
  const pivot = xToBeat(cursorX);
  store.view.px_per_beat = clamp(store.view.px_per_beat * factor, PX_MIN, PX_MAX);
  store.view.scroll_beat = pivot - cursorX / store.view.px_per_beat;
  clampView();
}

// ---------- culling: notes whose [start,end] intersect the visible beat window ----------
export function visibleNotes(padBeat = 0){
  const [b0, b1] = visibleBeatRange();
  const s = notesSorted();
  if (!s.length) return [];
  let maxLen = 1;
  for (const n of s) maxLen = Math.max(maxLen, n.len_beats);
  const lb = b0 - maxLen - padBeat;
  let lo = 0, hi = s.length;
  while (lo < hi){ const m = (lo + hi) >> 1; if (s[m].start_beat < lb) lo = m + 1; else hi = m; }
  const out = [];
  for (let i = lo; i < s.length; i++){
    const n = s[i];
    if (n.start_beat > b1 + padBeat) break;
    if (noteEnd(n) >= b0 - padBeat) out.push(n);
  }
  return out;
}
