// state.js — single app store, command-pattern undo/redo (cap 500), multi-selection
// (a Set of note ids), sorted-notes cache, id allocation, and shared helpers.
// Notes are stored in BEATS (tempo-invariant); seconds only exist at the audio boundary.

export const store = {
  config: null,           // GET /api/config
  models: [],             // GET /api/models
  projects: [],           // GET /api/projects
  projectId: null,
  doc: null,              // project.json document (canonical), incl. rev / bpm / time_sig / notes

  selection: new Set(),   // selected note ids (multi-select)
  tool: 'draw',           // draw | select | delete | slice
  lane: 'velocity',       // control-lane mode: velocity | pan | pitch
  snap: 'step',           // snap.js key
  loop: { enabled: false, start_beat: 0, end_beat: 4 },
  clipboard: null,        // { notes:[{dBeat,pitch,len,velocity,technique,bend,vibrato}], minBeat }

  view: { px_per_beat: 48, scroll_beat: 0, top_pitch: 96 },
  playhead: 0,            // beats
  lastLen: 1,             // last-used note length in beats (for the draw tool)

  follow: true,           // follow-playhead during playback
  master: 1,              // master volume 0..1
  altFree: false,         // alt held => temporary free (no snap)

  save: { state: 'saved', err: null },   // saved | saving | unsaved | error
  drag: null,             // live drag descriptor for the overlay
  hover: null,            // { beat, pitch }

  // render freshness: fresh=true only while the loaded render matches the doc.
  // Any doc mutation (see markDirty) flips fresh=false so playback falls back to
  // the preview synth and the waveform lane dims. meta = last render's meta.json.
  renderInfo: { fresh: false, meta: null },
};

// ---------- convenience getters ----------
export function bpm(){ return (store.doc && store.doc.bpm) || 140; }
export function timeSig(){ return (store.doc && store.doc.time_sig) || { num: 4, den: 4 }; }
export function notes(){ return (store.doc && store.doc.notes) || []; }
export const noteEnd = (n) => n.start_beat + n.len_beats;

// ---------- pub/sub ----------
const subs = new Set();
export function subscribe(fn){ subs.add(fn); return () => subs.delete(fn); }
export function notify(){ for (const fn of subs) fn(); }

// ---------- undo/redo (command pattern) ----------
// command = { label, mutate(doc), invert(doc), selAfter?, phAfter? }.  selAfter is an
// array/iterable of ids (or undefined => keep the pre-command selection).
const undoStack = [];
const redoStack = [];
const CAP = 500;

let sortedDirty = true;
let sortedCache = [];

export function apply(cmd){
  const selBefore = [...store.selection], phBefore = store.playhead;
  cmd.mutate(store.doc);
  invalidateSorted();
  const entry = {
    cmd, selBefore, phBefore,
    selAfter: cmd.selAfter !== undefined ? [...cmd.selAfter] : selBefore,
    phAfter:  cmd.phAfter  !== undefined ? cmd.phAfter  : phBefore,
  };
  setSelectionSilent(entry.selAfter);
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
  invalidateSorted();
  setSelectionSilent(e.selBefore);
  store.playhead = e.phBefore;
  redoStack.push(e);
  onDocChanged();
  return true;
}

export function redo(){
  const e = redoStack.pop();
  if (!e) return false;
  e.cmd.mutate(store.doc);
  invalidateSorted();
  setSelectionSilent(e.selAfter);
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

// ---------- mutation hook (render-staleness) ----------
// Fired from markDirty() — the single choke point every doc mutation passes through
// (note edits via apply/undo/redo -> onDocChanged, bpm/time-sig via transport). rendermgr
// registers here to invalidate a fresh render the instant the document changes.
const mutationHandlers = new Set();
export function onMutation(fn){ mutationHandlers.add(fn); return () => mutationHandlers.delete(fn); }

function onDocChanged(){ markDirty(); notify(); }
export function markDirty(){
  if (store.save.state !== 'saving') store.save.state = 'unsaved';
  if (store.renderInfo) store.renderInfo.fresh = false;
  for (const fn of mutationHandlers) fn();
  if (dirtyHandler) dirtyHandler();
}

// ---------- selection (non-undoable) ----------
function setSelectionSilent(ids){ store.selection = new Set(ids); }
export function setSelection(ids, { notify: doNotify = true } = {}){
  store.selection = new Set(ids);
  if (doNotify) notify();
}
export function selectOnly(id){ setSelection(id == null ? [] : [id]); }
export function addToSelection(id){ store.selection.add(id); notify(); }
export function toggleSelection(id){
  if (store.selection.has(id)) store.selection.delete(id); else store.selection.add(id);
  notify();
}
export function clearSelection(){ store.selection.clear(); notify(); }
export function isSelected(id){ return store.selection.has(id); }
export function selectAll(){ setSelection(notes().map(n => n.id)); }
export function selectedIds(){ return [...store.selection]; }
export function setPlayhead(b){ store.playhead = b; notify(); }

// ---------- note accessors ----------
export function notesSorted(){
  if (sortedDirty){
    sortedCache = notes().slice().sort((a, b) => a.start_beat - b.start_beat || a.id.localeCompare(b.id));
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

// ---------- id allocation (monotonic, never reused within a session) ----------
let idCounter = 0;
export function initIdCounter(doc){
  let max = -1;
  for (const n of (doc.notes || [])){
    const m = /^n(\d+)$/.exec(n.id);
    if (m) max = Math.max(max, +m[1]);
  }
  idCounter = Math.max((doc.next_note_ordinal | 0), max + 1);
}
export function allocId(){ return 'n' + String(idCounter++).padStart(4, '0'); }
export function nextOrdinal(){ return idCounter; }

// ---------- generic helpers ----------
export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const midiToHz = (m) => 440 * Math.pow(2, (m - 69) / 12);

const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
export function pitchName(m){
  m = Math.round(m);
  return NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
}
export const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
export function isBlackKey(m){ return BLACK_KEYS.has(((m % 12) + 12) % 12); }

// beats -> "bar:beat:tick" using the current time signature
export function barBeatTick(beat){
  const num = timeSig().num || 4;
  const b = Math.max(0, beat);
  const bar = Math.floor(b / num) + 1;
  const inBar = b - (bar - 1) * num;
  const beatIn = Math.floor(inBar) + 1;
  const tick = Math.round((inBar - Math.floor(inBar)) * 100);
  return `${bar}:${beatIn}:${String(tick).padStart(2, '0')}`;
}
