// editing.js — command factories. Each returns { label, mutate(doc), invert(doc),
// selAfter?, phAfter? } consumed by state.apply(). Factories read current values at
// construction time so create-then-apply immediately. All operate on arrays of ids so
// multi-selection is first-class. Every inverse is exact.
import { store, noteById, allocId, clamp } from './state.js';

const MIN_LEN = 0.03125;   // 1/128 note floor

export function makeNote({ start_beat, len_beats, pitch, velocity = 100, technique = 'normal', pan = 0 }){
  return {
    id: allocId(),
    start_beat, len_beats, pitch,
    velocity: clamp(Math.round(velocity), 1, 127),
    technique,
    pan: clamp(pan, -1, 1),
    bend: [{ beat: 0.0, semitones: 0.0 }],
    vibrato: { depth_semitones: 0.0, rate_hz: 5.5, onset_beats: 0.15 },
  };
}

// ---------- create ----------
export function createNotes(noteObjs){
  const arr = noteObjs.slice();
  const ids = arr.map(n => n.id);
  return {
    label: 'create', selAfter: ids, phAfter: arr.length ? arr[0].start_beat : undefined,
    mutate(doc){ for (const n of arr) doc.notes.push(n); },
    invert(doc){ const s = new Set(ids); doc.notes = doc.notes.filter(n => !s.has(n.id)); },
  };
}
// convenience single-note create
export function createNote(spec){ return createNotes([makeNote(spec)]); }

// ---------- delete ----------
export function deleteNotes(ids){
  const set = new Set(ids);
  // snapshot with original index for a faithful inverse ordering
  const snaps = [];
  store.doc.notes.forEach((n, i) => { if (set.has(n.id)) snaps.push({ i, n }); });
  return {
    label: 'delete', selAfter: [],
    mutate(doc){ doc.notes = doc.notes.filter(n => !set.has(n.id)); },
    invert(doc){ for (const { i, n } of snaps) doc.notes.splice(Math.min(i, doc.notes.length), 0, n); },
  };
}

// ---------- move (beat + pitch), snapped by caller ----------
export function moveNotes(ids, dBeat, dPitch, opts = {}){
  const old = ids.map(id => { const n = noteById(id); return { id, start: n.start_beat, pitch: n.pitch }; });
  return {
    label: 'move', selAfter: ids, phAfter: opts.phAfter,
    mutate(){ for (const o of old){ const n = noteById(o.id); n.start_beat = Math.max(0, o.start + dBeat); n.pitch = clamp(o.pitch + dPitch, 0, 127); } },
    invert(){ for (const o of old){ const n = noteById(o.id); n.start_beat = o.start; n.pitch = o.pitch; } },
  };
}

// ---------- resize (drag an edge) ----------
export function resizeNotes(ids, edge, dBeat){
  const old = ids.map(id => { const n = noteById(id); return { id, start: n.start_beat, len: n.len_beats }; });
  return {
    label: 'resize', selAfter: ids,
    mutate(){
      for (const o of old){
        const n = noteById(o.id);
        if (edge === 'start'){
          const ns = Math.min(Math.max(0, o.start + dBeat), o.start + o.len - MIN_LEN);
          n.start_beat = ns; n.len_beats = (o.start + o.len) - ns;
        } else {
          n.len_beats = Math.max(MIN_LEN, o.len + dBeat);
        }
      }
    },
    invert(){ for (const o of old){ const n = noteById(o.id); n.start_beat = o.start; n.len_beats = o.len; } },
  };
}

// ---------- velocity ----------
export function setVelocity(ids, vel){
  const list = Array.isArray(ids) ? ids : [ids];
  const old = list.map(id => ({ id, v: noteById(id).velocity }));
  const nv = clamp(Math.round(vel), 1, 127);
  return {
    label: 'velocity', selAfter: list,
    mutate(){ for (const o of old) noteById(o.id).velocity = nv; },
    invert(){ for (const o of old) noteById(o.id).velocity = o.v; },
  };
}
// per-note velocity map (for lane drag-paint ramps): {id: vel}
export function setVelocities(map){
  const ids = Object.keys(map);
  const old = ids.map(id => ({ id, v: noteById(id).velocity }));
  return {
    label: 'velocity', selAfter: ids,
    mutate(){ for (const id of ids) noteById(id).velocity = clamp(Math.round(map[id]), 1, 127); },
    invert(){ for (const o of old) noteById(o.id).velocity = o.v; },
  };
}

// ---------- pan (bipolar, -1..+1) ----------
// per-note pan map (for the Pan control-lane drag-paint): {id: pan}
export function setPans(map){
  const ids = Object.keys(map);
  const old = ids.map(id => ({ id, v: noteById(id).pan || 0 }));
  return {
    label: 'pan', selAfter: ids,
    mutate(){ for (const id of ids) noteById(id).pan = clamp(map[id], -1, 1); },
    invert(){ for (const o of old) noteById(o.id).pan = o.v; },
  };
}

// ---------- flat pitch bend (single control point, Pitch control-lane) ----------
// per-note semitone map: writes bend = [{beat:0, semitones:v}] on notes that have no
// multi-point curve. {id: semitones}
export function setBendsFlat(map){
  const ids = Object.keys(map);
  const old = ids.map(id => ({ id, b: JSON.parse(JSON.stringify(noteById(id).bend || [])) }));
  return {
    label: 'bend', selAfter: ids,
    mutate(){ for (const id of ids) noteById(id).bend = [{ beat: 0.0, semitones: clamp(map[id], -2, 2) }]; },
    invert(){ for (const o of old) noteById(o.id).bend = JSON.parse(JSON.stringify(o.b)); },
  };
}

// ---------- slice: split a note at an absolute beat ----------
export function sliceNote(id, atBeat){
  const n = noteById(id);
  if (!n) return null;
  const rel = atBeat - n.start_beat;
  if (rel <= MIN_LEN || rel >= n.len_beats - MIN_LEN) return null;
  const oldLen = n.len_beats;
  const right = {
    id: allocId(), start_beat: atBeat, len_beats: oldLen - rel, pitch: n.pitch,
    velocity: n.velocity, technique: n.technique, pan: n.pan || 0,
    bend: JSON.parse(JSON.stringify(n.bend || [{ beat: 0, semitones: 0 }])),
    vibrato: JSON.parse(JSON.stringify(n.vibrato || { depth_semitones: 0, rate_hz: 5.5, onset_beats: 0.15 })),
  };
  return {
    label: 'slice', selAfter: [id, right.id], phAfter: atBeat,
    mutate(doc){ const m = noteById(id); m.len_beats = rel; doc.notes.push(right); },
    invert(doc){ const m = noteById(id); m.len_beats = oldLen; doc.notes = doc.notes.filter(x => x.id !== right.id); },
  };
}

// ---------- paste (create a batch, select it) ----------
export function paste(noteObjs){ return createNotes(noteObjs); }

// ======================================================================
// Phase-2 stubs — functional but the UI that drives them lands in Phase 2.
// Signatures are fixed here so Phase 2 can wire buttons without touching apply().
// ======================================================================
export function setTechnique(ids, techName){
  const list = Array.isArray(ids) ? ids : [ids];
  const old = list.map(id => ({ id, t: noteById(id).technique }));
  return {
    label: 'technique', selAfter: list,
    mutate(){ for (const o of old) noteById(o.id).technique = techName; },
    invert(){ for (const o of old) noteById(o.id).technique = o.t; },
  };
}
export function setBend(id, bendPoints){
  const n = noteById(id);
  const old = JSON.parse(JSON.stringify(n.bend || []));
  const next = JSON.parse(JSON.stringify(bendPoints));
  return {
    label: 'bend', selAfter: [id],
    mutate(){ noteById(id).bend = JSON.parse(JSON.stringify(next)); },
    invert(){ noteById(id).bend = JSON.parse(JSON.stringify(old)); },
  };
}
// Merge a partial vibrato patch (e.g. {depth_semitones: v}) into every target note,
// preserving each note's other fields. `ids` may be a single id or an array.
const DEF_VIB = { depth_semitones: 0.0, rate_hz: 5.5, onset_beats: 0.15 };
export function setVibrato(ids, patch){
  const list = Array.isArray(ids) ? ids : [ids];
  const old = list.map(id => ({ id, v: JSON.parse(JSON.stringify(noteById(id).vibrato || DEF_VIB)) }));
  const p = JSON.parse(JSON.stringify(patch));
  return {
    label: 'vibrato', selAfter: list,
    mutate(){ for (const o of old){ const n = noteById(o.id); n.vibrato = Object.assign({}, DEF_VIB, n.vibrato, p); } },
    invert(){ for (const o of old) noteById(o.id).vibrato = JSON.parse(JSON.stringify(o.v)); },
  };
}
