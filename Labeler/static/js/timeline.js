// timeline.js — time<->pixel and pitch<->pixel transforms, shared layout geometry,
// pan/zoom, and note culling. All canvases share one horizontal time mapping.
import { store, clamp, notesSorted } from './state.js';

export const PX_MIN = 40, PX_MAX = 1000, ZOOM_STEP = 1.25;
const NOTE_BINS = 1.5;                // note height in mel bins (spec)
// mel bin count from server meta (BigVGAN N_MELS, normally 128).
export function binCount(){ return (store.meta && store.meta.mel && store.meta.mel.n_mels) || 128; }

// Live pixel geometry, filled by render.resize(). Bands are y-offsets within .roll.
export const layout = {
  dpr: 1,
  width: 800,           // shared drawing width (CSS px)
  rulerTop: 0, rulerH: 22,
  rollTop: 22, rollH: 300,      // pianoroll band (mel/notes)
  velTop: 322, velH: 70,
  onsetTop: 392, onsetH: 48,
  totalH: 440,
  mmWidth: 800, mmHeight: 46,   // minimap
};

export function durationS(){
  return (store.meta && store.meta.duration_s) || 30;
}

// ---------- time <-> x (shared piano-roll / ruler / lanes) ----------
export const timeToX = (t) => (t - store.view.originS) * store.view.pxPerSec;
export const xToTime = (x) => store.view.originS + x / store.view.pxPerSec;
export function visibleTimeRange(){
  const t0 = store.view.originS;
  return [t0, t0 + layout.width / store.view.pxPerSec];
}

// ---------- mel bin <-> y (origin lower: bin 0 at bottom) ----------
export const binToY = (bin) => layout.rollH * (1 - bin / (binCount() - 1));
export const yToBin = (y) => (1 - y / layout.rollH) * (binCount() - 1);

// mel bin of a (fractional) midi pitch, from server meta; fallback to a mel curve.
export function binForPitch(pitch){
  const mb = store.meta && store.meta.mel && store.meta.mel_bin_of_midi
           ? store.meta.mel_bin_of_midi
           : (store.meta && store.meta.mel && store.meta.mel.mel_bin_of_midi);
  const arr = mb || null;
  if (arr){
    const lo = Math.floor(pitch), hi = Math.ceil(pitch);
    if (lo === hi) return arr[clamp(lo, 0, 127)];
    const a = arr[clamp(lo, 0, 127)], b = arr[clamp(hi, 0, 127)];
    return a + (b - a) * (pitch - lo);
  }
  // fallback mel mapping across [0, sr/2]
  const sr = (store.config && store.config.sr) || 44100;
  const hz = 440 * Math.pow(2, (pitch - 69) / 12);
  const mel = (f) => 2595 * Math.log10(1 + f / 700);
  return (mel(hz) / mel(sr / 2)) * (binCount() - 1);
}

// invert bin->nearest integer midi (violin-ish search range)
export function pitchForBin(bin){
  let best = 60, bestd = Infinity;
  for (let m = 40; m <= 100; m++){
    const d = Math.abs(binForPitch(m) - bin);
    if (d < bestd){ bestd = d; best = m; }
  }
  return best;
}

// ---------- note rectangles (used by render AND interact hit-testing) ----------
export function noteRect(note){
  const x = timeToX(note.start_s);
  const w = Math.max(2, (note.end_s - note.start_s) * store.view.pxPerSec);
  const yc = binToY(binForPitch(note.pitch));
  const h = Math.max(3, NOTE_BINS * (layout.rollH / (binCount() - 1)));
  return { x, y: yc - h / 2, w, h };
}

export function velBarRect(note){
  const x = timeToX(note.start_s);
  const w = Math.max(2, (note.end_s - note.start_s) * store.view.pxPerSec);
  const frac = clamp(note.velocity / 127, 0, 1);
  const h = frac * (layout.velH - 4);
  return { x, y: layout.velH - h, w, h };
}

// ---------- pan / zoom (not undoable) ----------
export function clampView(){
  const dur = durationS();
  const viewW = layout.width / store.view.pxPerSec;
  const maxOrigin = Math.max(0, dur - viewW * 0.5); // allow a little scroll past end
  store.view.originS = clamp(store.view.originS, -0.05, maxOrigin);
}
export function panBy(dtSec){ store.view.originS += dtSec; clampView(); }
export function zoomAbout(cursorX, factor){
  const pivot = xToTime(cursorX);
  store.view.pxPerSec = clamp(store.view.pxPerSec * factor, PX_MIN, PX_MAX);
  store.view.originS = pivot - cursorX / store.view.pxPerSec;
  clampView();
}

// ---------- culling: notes whose [start,end] intersect the visible window ----------
export function visibleNotes(padSec = 0){
  const [t0, t1] = visibleTimeRange();
  const s = notesSorted();
  if (!s.length) return [];
  let maxDur = 0.5;
  for (const n of s) maxDur = Math.max(maxDur, n.end_s - n.start_s);
  // first index with start_s >= t0 - maxDur
  let lo = 0, hi = s.length;
  const lb = t0 - maxDur - padSec;
  while (lo < hi){ const m = (lo + hi) >> 1; if (s[m].start_s < lb) lo = m + 1; else hi = m; }
  const out = [];
  for (let i = lo; i < s.length; i++){
    const n = s[i];
    if (n.start_s > t1 + padSec) break;
    if (n.end_s >= t0 - padSec) out.push(n);
  }
  return out;
}
