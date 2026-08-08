// state.js — single app store, command-pattern undo/redo, and small shared helpers.
// (Shared helpers live here rather than a separate util.js to keep the module set
//  exactly as specified in the plan.)

export const store = {
  config: null,          // GET /api/config
  clips: [],             // GET /api/clips
  clipId: null,
  meta: null,            // GET /api/clips/{id}/meta
  doc: null,             // notes.json document (canonical), incl. rev
  onset: null,           // Float32Array of onset strength (hop 512)

  selectedId: null,
  playhead: 0,           // seconds (clip time)
  view: { pxPerSec: 200, originS: 0 },

  speed: 1,              // 1 | 0.5 | 0.25
  variant: 'cleaned',    // 'source' (raw) | 'cleaned'
  follow: true,          // auto-pan during playback
  paintMode: false,      // mute-region paint armed (g)

  tracks: { orig: { vol: 1, mute: false }, midi: { vol: 0.8, mute: false } },

  save: { state: 'saved', err: null },   // saved | saving | unsaved | error
  processing: null,      // last status object while a job runs
  hover: null,           // {t, pitch}
  drag: null,            // live drag descriptor for overlay ghost
  decoding: false,       // waiting on a needed audio buffer
};

// ---------- pub/sub ----------
const subs = new Set();
export function subscribe(fn){ subs.add(fn); return () => subs.delete(fn); }
export function notify(){ for (const fn of subs) fn(); }

// ---------- undo/redo (command pattern) ----------
// A command = { label, mutate(doc), invert(doc), selAfter?, phAfter? }.
// The store captures selection+playhead *before* automatically; commands may
// declare the resulting selection/playhead (e.g. label-key auto-advance).
const undoStack = [];
const redoStack = [];
const CAP = 500;

let sortedDirty = true;
let sortedCache = [];

export function apply(cmd){
  const selBefore = store.selectedId, phBefore = store.playhead;
  cmd.mutate(store.doc);
  const entry = {
    cmd, selBefore, phBefore,
    selAfter: cmd.selAfter !== undefined ? cmd.selAfter : selBefore,
    phAfter:  cmd.phAfter  !== undefined ? cmd.phAfter  : phBefore,
  };
  store.selectedId = entry.selAfter;
  store.playhead = entry.phAfter;
  undoStack.push(entry);
  if (undoStack.length > CAP) undoStack.shift();
  redoStack.length = 0;
  onDocChanged();
}

export function undo(){
  const e = undoStack.pop();
  if (!e) return false;
  e.cmd.invert(store.doc);
  store.selectedId = e.selBefore;
  store.playhead = e.phBefore;
  redoStack.push(e);
  onDocChanged();
  return true;
}

export function redo(){
  const e = redoStack.pop();
  if (!e) return false;
  e.cmd.mutate(store.doc);
  store.selectedId = e.selAfter;
  store.playhead = e.phAfter;
  undoStack.push(e);
  onDocChanged();
  return true;
}

export function canUndo(){ return undoStack.length > 0; }
export function canRedo(){ return redoStack.length > 0; }
export function clearHistory(){ undoStack.length = 0; redoStack.length = 0; }

// ---------- dirty / autosave hook ----------
let dirtyHandler = null;
export function setDirtyHandler(fn){ dirtyHandler = fn; }

function onDocChanged(){
  sortedDirty = true;
  markDirty();
  notify();
}
export function markDirty(){
  if (store.save.state !== 'saving') store.save.state = 'unsaved';
  if (dirtyHandler) dirtyHandler();
}

// ---------- selection / playhead (non-undoable navigation) ----------
export function select(id, { playhead } = {}){
  store.selectedId = id;
  if (playhead !== undefined) store.playhead = playhead;
  notify();
}
export function setPlayhead(t){ store.playhead = t; notify(); }
export function deselect(){ store.selectedId = null; notify(); }

// ---------- note accessors ----------
export function notes(){ return (store.doc && store.doc.notes) || []; }

export function notesSorted(){
  if (sortedDirty){
    sortedCache = notes().slice().sort((a, b) => a.start_s - b.start_s || a.id.localeCompare(b.id));
    sortedDirty = false;
  }
  return sortedCache;
}
export function invalidateSorted(){ sortedDirty = true; }

export function noteById(id){
  const ns = notes();
  for (let i = 0; i < ns.length; i++) if (ns[i].id === id) return ns[i];
  return null;
}

export function nextNote(id){
  const s = notesSorted();
  const i = s.findIndex(n => n.id === id);
  return (i >= 0 && i < s.length - 1) ? s[i + 1] : null;
}
export function prevNote(id){
  const s = notesSorted();
  const i = s.findIndex(n => n.id === id);
  return (i > 0) ? s[i - 1] : null;
}
export function firstNote(){ const s = notesSorted(); return s[0] || null; }

// ---------- id allocation (monotonic, never reused within a session) ----------
let idCounter = 0;
export function initIdCounter(doc){
  let max = -1;
  for (const n of (doc.notes || [])){
    const m = /^n(\d+)$/.exec(n.id);
    if (m) max = Math.max(max, +m[1]);
  }
  // server tracks next_note_ordinal; stay ahead of both it and any existing id.
  idCounter = Math.max((doc.next_note_ordinal | 0), max + 1);
}
export function allocId(){ return 'n' + String(idCounter++).padStart(4, '0'); }
export function nextOrdinal(){ return idCounter; }

// ---------- generic helpers ----------
export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const midiToHz = (m) => 440 * Math.pow(2, (m - 69) / 12);

// ---------- per-note energy envelope (shared by render + synth) ----------
// Reconstruct the A-weighted loudness curve db(u) at normalized position u in 0..1
// from env_dct [c0,c1,c2]:  db(u) = c0 + c1*cos(PI*u) + c2*cos(2*PI*u)
//   (c0 level, c1 tilt, c2 arch).  Notes lacking env_dct fall back to a flat level
// derived from velocity; VEL_C0_A/VEL_C0_B mirror common/envelope.py VEL_C0_A / VEL_C0_B
// (UI-anchored map: vel 1..127 -> -30..+16 dB, the envelope lanes' display range).
export const VEL_C0_A = 46 / 126, VEL_C0_B = -30 - VEL_C0_A;
export const envDb = (c, u) => c[0] + c[1] * Math.cos(Math.PI * u) + c[2] * Math.cos(2 * Math.PI * u);
export function noteEnvCoeffs(n){
  return (Array.isArray(n.env_dct) && n.env_dct.length >= 3)
    ? n.env_dct
    : [VEL_C0_A * n.velocity + VEL_C0_B, 0, 0];
}

// ---------- per-note vibrato params (render-only, for now) ----------
// Notes flagged `vibrato` may also carry `vib_params` (fitted oscillator numbers) and
// `vib_model` (the name of the model that reads them) once POST /vibrato-fit has run;
// for the current `rampboth5` that is [D0, gD, f0, s, phi] — half-depth cents, cents/s,
// Hz, Hz/s, radians, all anchored at the note ONSET (see common/vibrato_model.py).
// They are onset-anchored, so editing.js clears them on any boundary/pitch edit.
// render.js scales the vibrato squiggle from them; the preview synth ignores them.

const NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
export function pitchName(m){
  m = Math.round(m);
  return NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
}

export function fmtTime(s){
  if (!isFinite(s)) s = 0;
  const sign = s < 0 ? '-' : ''; s = Math.abs(s);
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${sign}${m}:${sec.toFixed(3).padStart(6, '0')}`;
}

// dotted-path get/set (supports "human.technique")
export function getPath(obj, path){
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
}
export function setPath(obj, path, val){
  const ks = path.split('.');
  let o = obj;
  for (let i = 0; i < ks.length - 1; i++){
    if (o[ks[i]] == null || typeof o[ks[i]] !== 'object') o[ks[i]] = {};
    o = o[ks[i]];
  }
  o[ks[ks.length - 1]] = val;
}
