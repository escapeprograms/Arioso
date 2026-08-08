// editing.js — command factories. Each returns { label, mutate, invert, selAfter?, phAfter? }
// consumed by state.apply(). Factories read current values at construction time to
// build a correct inverse, so create-then-apply immediately.
import { store, noteById, getPath, setPath, allocId, clamp } from './state.js';

const MIN_DUR = 0.02;

function newHuman(){ return { technique: false, vibrato: false, velocity: false, timing: false }; }

// --- vibrato params are onset-anchored (see common/vibrato_model.py) ---
// vib_params describe cents-vs-seconds *since the note onset*, measured against the
// note's scored pitch, and the depth/rate ramps extrapolate badly outside the span
// they were fitted on. So any edit that moves a boundary or the pitch — or that
// clears the vibrato flag the params belong to — drops them; the next vibrato-fit
// pass re-measures. Every such command snapshots them first so undo restores them.
function vibSnap(n){ return { p: n.vib_params, m: n.vib_model }; }
function vibClear(n){ n.vib_params = null; n.vib_model = null; }
function vibRestore(n, s){ n.vib_params = s.p; n.vib_model = s.m; }

// --- generic field patch (dotted paths, e.g. 'human.technique') ---
export function setFields(id, fields, opts = {}){
  const note = noteById(id);
  const old = {};
  for (const k of Object.keys(fields)) old[k] = getPath(note, k);
  return {
    label: opts.label || 'edit',
    selAfter: opts.selAfter, phAfter: opts.phAfter,
    mutate(){ const n = noteById(id); for (const k in fields) setPath(n, k, fields[k]); },
    invert(){ const n = noteById(id); for (const k in old) setPath(n, k, old[k]); },
  };
}

export function setTechnique(id, techName, opts = {}){
  return setFields(id, { technique: techName, 'human.technique': true },
                   { label: 'technique', ...opts });
}
// Turning vibrato OFF also drops the fitted params (they described an oscillation the
// label now says is not there); turning it ON leaves them alone — a note that is
// re-flagged has none until the next fit pass.
export function setVibrato(id, on){
  const fields = { vibrato: !!on, 'human.vibrato': true };
  if (!on) Object.assign(fields, { vib_params: null, vib_model: null });
  return setFields(id, fields, { label: 'vibrato' });
}
export function toggleVibrato(id){
  return setVibrato(id, !noteById(id).vibrato);
}
export function setVelocity(id, vel){
  return setFields(id, { velocity: clamp(Math.round(vel), 1, 127), 'human.velocity': true },
                   { label: 'velocity' });
}

// --- move / re-pitch (drag body) ---
export function moveNote(id, dStart, dPitch, opts = {}){
  const n = noteById(id);
  const o = { start: n.start_s, end: n.end_s, pitch: n.pitch, timing: n.human.timing };
  const vib = vibSnap(n);
  const newPitch = clamp(Math.round(o.pitch + dPitch), 0, 127);
  return {
    label: 'move', selAfter: id, phAfter: opts.phAfter,
    mutate(){ const m = noteById(id); m.start_s = o.start + dStart; m.end_s = o.end + dStart; m.pitch = newPitch; m.human.timing = true; vibClear(m); },
    invert(){ const m = noteById(id); m.start_s = o.start; m.end_s = o.end; m.pitch = o.pitch; m.human.timing = o.timing; vibRestore(m, vib); },
  };
}

// --- resize (drag right edge) ---
export function resizeNote(id, dSec){
  const n = noteById(id);
  const o = { start: n.start_s, end: n.end_s, timing: n.human.timing };
  const vib = vibSnap(n);
  const ne = Math.max(o.end + dSec, o.start + MIN_DUR);
  return {
    label: 'resize', selAfter: id,
    mutate(){ const m = noteById(id); m.end_s = ne; m.human.timing = true; vibClear(m); },
    invert(){ const m = noteById(id); m.end_s = o.end; m.human.timing = o.timing; vibRestore(m, vib); },
  };
}

// --- add note ---
export function addNote({ start_s, end_s, pitch, technique, velocity = 80 }){
  const id = allocId();
  const note = {
    id, start_s, end_s, pitch, velocity, technique, vibrato: false, slur_group: null,
    auto: {}, human: { technique: false, vibrato: false, velocity: false, timing: true },
  };
  return {
    label: 'add', selAfter: id, phAfter: start_s,
    mutate(doc){ doc.notes.push(note); },
    invert(doc){ const i = doc.notes.findIndex(x => x.id === id); if (i >= 0) doc.notes.splice(i, 1); },
  };
}

// --- delete note ---
export function deleteNote(id){
  const idx = store.doc.notes.findIndex(x => x.id === id);
  const snap = idx >= 0 ? store.doc.notes[idx] : null;
  return {
    label: 'delete', selAfter: null,
    mutate(doc){ const i = doc.notes.findIndex(x => x.id === id); if (i >= 0) doc.notes.splice(i, 1); },
    invert(doc){ if (snap) doc.notes.splice(Math.min(idx, doc.notes.length), 0, snap); },
  };
}

// --- split note at time t (labels copied; both flagged human.timing) ---
export function splitNote(id, t){
  const n = noteById(id);
  if (!n || t <= n.start_s + MIN_DUR || t >= n.end_s - MIN_DUR) return null;
  const oldEnd = n.end_s, oldTiming = n.human.timing;
  const vib = vibSnap(n);
  const rightId = allocId();
  // The right half starts at t, so the left half's onset-anchored params describe
  // neither piece: both sides come out unfitted.
  const right = {
    id: rightId, start_s: t, end_s: oldEnd, pitch: n.pitch, velocity: n.velocity,
    technique: n.technique, vibrato: n.vibrato, slur_group: n.slur_group,
    vib_params: null, vib_model: null,
    auto: { ...(n.auto || {}) },
    human: { ...n.human, timing: true },
  };
  return {
    label: 'split', selAfter: id, phAfter: t,
    mutate(doc){ const m = noteById(id); m.end_s = t; m.human.timing = true; vibClear(m); doc.notes.push(right); },
    invert(doc){ const m = noteById(id); m.end_s = oldEnd; m.human.timing = oldTiming; vibRestore(m, vib);
                 const i = doc.notes.findIndex(x => x.id === rightId); if (i >= 0) doc.notes.splice(i, 1); },
  };
}

// --- split every note containing time t (one undo step) ---
export function splitAllAt(t){
  const cmds = store.doc.notes.map(n => splitNote(n.id, t)).filter(Boolean);
  if (!cmds.length) return null;
  return {
    label: 'split', selAfter: store.selectedId, phAfter: t,
    mutate(doc){ for (const c of cmds) c.mutate(doc); },
    invert(doc){ for (let i = cmds.length - 1; i >= 0; i--) cmds[i].invert(doc); },
  };
}

// --- mute regions ---
export function addMuteRegion(region){
  const r = { start_s: Math.min(region.start_s, region.end_s),
              end_s: Math.max(region.start_s, region.end_s),
              label: region.label || '' };
  return {
    label: 'mute-region',
    mutate(doc){ (doc.mute_regions || (doc.mute_regions = [])).push(r); },
    invert(doc){ const i = doc.mute_regions.indexOf(r); if (i >= 0) doc.mute_regions.splice(i, 1); },
  };
}
export function deleteMuteRegion(index){
  const snap = store.doc.mute_regions[index];
  return {
    label: 'del-region',
    mutate(doc){ doc.mute_regions.splice(index, 1); },
    invert(doc){ doc.mute_regions.splice(index, 0, snap); },
  };
}

export { newHuman };
