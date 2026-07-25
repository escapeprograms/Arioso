// clipboard.js — copy / cut / paste for multi-selection. The clipboard stores notes
// beat-relative to the earliest selected note (pitch absolute). Paste lands the batch at
// the playhead, preserving relative onsets, lengths, and pitches. copy() returns a boolean;
// cut() and paste() return the count of notes affected (0 = nothing / empty clipboard).
import { store, apply, notesSorted, noteById, selectedIds } from './state.js';
import * as edit from './editing.js';

function snapshotSelection(){
  const ids = selectedIds();
  if (!ids.length) return null;
  const sel = ids.map(noteById).filter(Boolean);
  const minBeat = Math.min(...sel.map(n => n.start_beat));
  const notes = sel.map(n => {
    const t = {
      dBeat: n.start_beat - minBeat,
      len: n.len_beats, pitch: n.pitch, velocity: n.velocity, technique: n.technique, pan: n.pan || 0,
      bend: JSON.parse(JSON.stringify(n.bend || [{ beat: 0, semitones: 0 }])),
      vibrato: JSON.parse(JSON.stringify(n.vibrato || { depth_semitones: 0, rate_hz: 5.5, onset_beats: 0.15 })),
    };
    // carry the energy envelope only when present, so env-less notes stay env-less (segment hash stability)
    if (Array.isArray(n.env_dct) && n.env_dct.length >= 3) t.env_dct = n.env_dct.slice();
    return t;
  });
  return { notes, minBeat };
}

export function copy(){
  const snap = snapshotSelection();
  if (snap) store.clipboard = snap;
  return !!snap;
}

export function cut(){
  const ids = selectedIds();
  if (!ids.length) return 0;
  copy();
  apply(edit.deleteNotes(ids));
  return ids.length;
}

// Paste at the playhead (default) or at an explicit beat, preserving relative layout.
export function paste(atBeat = store.playhead){
  const cb = store.clipboard;
  if (!cb || !cb.notes.length) return 0;
  const base = Math.max(0, atBeat);
  const notes = cb.notes.map(t => {
    const n = edit.makeNote({
      start_beat: base + t.dBeat, len_beats: t.len, pitch: t.pitch,
      velocity: t.velocity, technique: t.technique, pan: t.pan || 0,
    });
    n.bend = JSON.parse(JSON.stringify(t.bend));
    n.vibrato = JSON.parse(JSON.stringify(t.vibrato));
    if (t.env_dct) n.env_dct = t.env_dct.slice();   // only pasted notes that were copied with an envelope
    return n;
  });
  apply(edit.paste(notes));
  return notes.length;
}

// re-exported for keymap wiring
export function hasClipboard(){ return !!(store.clipboard && store.clipboard.notes.length); }
export { notesSorted };
