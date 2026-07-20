// mock.js — in-browser fake backend (?mock=1 or file://) backed by localStorage, so the
// Studio frontend runs with zero backend. Mirrors the Studio/api.py contract shapes.
const LS_KEY = 'studio.mock.projects';
const DEFAULT_ID = 'my-song';

// Mirrors Studio/config.py _DEFAULT_ARTICULATIONS (name/key/color/abbrev/model_vocab).
const ARTICULATIONS = [
  { name: 'normal',   key: '1', color: '#6cc04a', abbrev: 'norm', model_vocab: 'normal'   },
  { name: 'slur',     key: '2', color: '#dd8452', abbrev: 'slur', model_vocab: 'normal'   },
  { name: 'spiccato', key: '3', color: '#4c9be8', abbrev: 'spic', model_vocab: 'spiccato' },
  { name: 'detache',  key: '4', color: '#c46bd0', abbrev: 'det',  model_vocab: 'normal'   },
];

// ---------- localStorage store ----------
function loadAll(){
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; }
}
function saveAll(map){
  try { localStorage.setItem(LS_KEY, JSON.stringify(map)); } catch {}
}

function demoNotes(){
  // a short violin phrase in beats (den=4 => beat = quarter note). A few notes carry
  // techniques / pan / bends / vibrato so Phase-2 rendering is visible at a glance.
  //          start len  pitch tech        pan   bend                                     vib(depth,rate,onset)
  const seq = [
    [0.0, 1.0, 67, 'normal',   0.0, null,                                              [0.0, 5.5, 0.15]],
    [1.0, 1.0, 69, 'slur',    -0.3, null,                                              [0.0, 5.5, 0.15]],
    [2.0, 1.0, 71, 'slur',    -0.3, null,                                              [0.0, 5.5, 0.15]],
    [3.0, 1.0, 72, 'spiccato', 0.2, null,                                              [0.0, 5.5, 0.15]],
    [4.0, 2.0, 74, 'normal',   0.0, [[0,-1],[0.5,0],[2,0]],                            [0.6, 5.5, 0.30]],
    [6.0, 1.0, 71, 'detache',  0.4, [[0, 0.5]],                                        [0.0, 5.5, 0.15]],
    [7.0, 1.0, 67, 'normal',   0.0, null,                                              [0.0, 5.5, 0.15]],
    [8.0, 0.5, 72, 'spiccato', 0.0, null,                                              [0.0, 5.5, 0.15]],
    [8.5, 0.5, 74, 'spiccato', 0.0, null,                                             [0.0, 5.5, 0.15]],
    [9.0, 1.0, 76, 'normal',   0.0, null,                                              [0.9, 6.5, 0.10]],
    [10.0, 2.0, 79, 'normal',  0.0, [[0,0],[1,2],[2,2]],                               [0.4, 5.0, 0.20]],
  ];
  return seq.map(([start, len, pitch, technique, pan, bend, vib], i) => ({
    id: 'n' + String(i).padStart(4, '0'),
    start_beat: start, len_beats: len, pitch,
    velocity: 80 + ((i * 7) % 40),
    technique,
    pan,
    bend: bend ? bend.map(([beat, semitones]) => ({ beat, semitones })) : [{ beat: 0.0, semitones: 0.0 }],
    vibrato: { depth_semitones: vib[0], rate_hz: vib[1], onset_beats: vib[2] },
  }));
}

function buildProject(id = DEFAULT_ID, name = 'My Song'){
  const notes = demoNotes();
  return {
    schema_version: 1, project_id: id, rev: 1,
    next_note_ordinal: notes.length,
    name, bpm: 140.0, time_sig: { num: 4, den: 4 }, ppq: 480,
    render: { model: '7-9-adr', checkpoint: 'checkpoint_final.pt', prior_mode: 'bend',
              wav: null, duration_s: 0, segments: [] },
    view: { px_per_beat: 48.0, scroll_beat: 0.0, top_pitch: 96, lane: 'velocity', snap: 'step' },
    notes,
  };
}

function ensureSeed(){
  const map = loadAll();
  if (!Object.keys(map).length){ map[DEFAULT_ID] = buildProject(); saveAll(map); }
  return map;
}

// ---------- API surface (consumed by api.js) ----------
export async function getConfig(){
  return {
    articulations: ARTICULATIONS,
    keys: Object.fromEntries(ARTICULATIONS.map(a => [a.key, a.name])),
    default_model: '7-9-adr',
    default_checkpoint: 'checkpoint_final.pt',
    default_prior_mode: 'bend',
    pitch_range: { min: 40, max: 100 },
    prior_modes: ['bend', 'quantized'],
  };
}

export async function getModels(){
  // Mirrors the real GET /api/models object shape (run/checkpoints/default + defaults).
  return {
    models: [
      { run: '7-9-adr', checkpoints: ['checkpoint_final.pt', 'checkpoint_100k.pt'], default: true },
      { run: '7-7-offline-aug', checkpoints: ['checkpoint_final.pt'], default: false },
    ],
    default_model: '7-9-adr',
    default_checkpoint: 'checkpoint_final.pt',
  };
}

export async function listProjects(){
  const map = ensureSeed();
  // Mirror the backend project_store.summary() shape (key `id`, not `project_id`).
  return Object.values(map).map(p => ({
    id: p.project_id, name: p.name, rev: p.rev,
    bpm: p.bpm, n_notes: (p.notes || []).length,
  }));
}

export async function createProject(body = {}){
  const map = loadAll();
  let id = (body.project_id || body.name || 'project')
    .toString().trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'project';
  let base = id, i = 1;
  while (map[id]) id = `${base}-${++i}`;
  const p = buildProject(id, body.name || id);
  p.notes = [];                 // fresh projects start empty
  p.next_note_ordinal = 0;
  map[id] = p; saveAll(map);
  return JSON.parse(JSON.stringify(p));
}

export async function getProject(id){
  const map = ensureSeed();
  const p = map[id] || map[DEFAULT_ID];
  return JSON.parse(JSON.stringify(p));
}

export async function putProject(id, doc){
  const map = loadAll();
  const cur = map[id];
  if (cur && doc.rev != null && doc.rev !== cur.rev){
    const err = new Error('rev_conflict'); err.status = 409; err.code = 'rev_conflict'; throw err;
  }
  const next = JSON.parse(JSON.stringify(doc));
  next.rev = ((cur && cur.rev) || doc.rev || 0) + 1;
  next.project_id = id;
  map[id] = next; saveAll(map);
  return { rev: next.rev };
}

// ---------- render (simulated) ----------
// A staged ~2s progress (serialize -> prior -> model -> vocoder -> write) then a fake meta
// whose wav + peaks are real data: URLs built client-side (a decaying-sine rendering of the
// project's notes) so the waveform lane AND the player source-switch are fully exercisable
// with no backend. Same contract as Studio/api.py: 409 while running, 404 until rendered.
const renders = {};   // id -> live { state, stage, pct, error, meta }

const STAGES = [['serialize', 6], ['prior', 26], ['model', 60], ['vocoder', 85], ['write', 96]];

export async function startRender(id, opts = {}){
  const cur = renders[id];
  if (cur && cur.state === 'processing'){
    const e = new Error('job_running'); e.status = 409; e.code = 'job_running'; throw e;
  }
  const proj = loadAll()[id] || buildProject(id);
  const st = { state: 'processing', stage: 'serialize', pct: 0, error: null, message: 'starting', meta: null };
  renders[id] = st;
  let step = 0;
  const advance = () => {
    if (st.state !== 'processing') return;
    if (step < STAGES.length){
      st.stage = STAGES[step][0]; st.pct = STAGES[step][1]; st.message = st.stage; step++;
      setTimeout(advance, 400);
    } else {
      try { st.meta = buildFakeRender(id, proj, opts); st.stage = 'write'; st.pct = 100; st.state = 'done'; st.message = 'done'; }
      catch (err){ st.state = 'error'; st.error = String((err && err.message) || err); }
    }
  };
  setTimeout(advance, 400);
  return { status: 'accepted', project_id: id, scope: opts.scope || 'phrase',
           model: opts.model, checkpoint: opts.checkpoint, prior_mode: opts.prior_mode };
}

export async function getRenderStatus(id){
  const st = renders[id];
  if (!st) return { state: 'idle', stage: null, pct: 0, error: null, message: '' };
  return { state: st.state, stage: st.stage, pct: st.pct, error: st.error, message: st.message };
}

export async function getRenderMeta(id){
  const st = renders[id];
  if (st && st.meta) return st.meta;
  const e = new Error('not_rendered'); e.status = 404; e.code = 'not_rendered'; throw e;
}

// ---------- fake audio synthesis (data: URL wav + peaks) ----------
const FAKE_SR = 8000, FAKE_BPS = 100;

function buildFakeRender(id, proj, opts){
  const bpm = proj.bpm || 140, spb = 60 / bpm;
  const notes = proj.notes || [];
  const sel = opts.scope === 'selection' && opts.note_ids ? new Set(opts.note_ids) : null;
  let endBeat = 0;
  for (const n of notes) endBeat = Math.max(endBeat, n.start_beat + n.len_beats);
  const durS = Math.max(1, endBeat * spb + 0.3);
  const N = Math.ceil(durS * FAKE_SR);
  const buf = new Float32Array(N);
  for (const n of notes){
    if (sel && !sel.has(n.id)) continue;
    const f0 = 440 * Math.pow(2, (n.pitch - 69) / 12);
    const i0 = Math.max(0, Math.floor(n.start_beat * spb * FAKE_SR));
    const i1 = Math.min(N, Math.floor((n.start_beat + n.len_beats) * spb * FAKE_SR));
    const amp = 0.25 * Math.pow((n.velocity || 90) / 127, 1.3);
    for (let i = i0; i < i1; i++){
      const tt = (i - i0) / FAKE_SR;
      const env = Math.min(1, tt / 0.01) * Math.exp(-tt * 2.0);
      buf[i] += amp * env * Math.sin(2 * Math.PI * f0 * (i / FAKE_SR));
    }
  }
  for (let i = 0; i < N; i++){ buf[i] = Math.max(-1, Math.min(1, buf[i])); }
  return {
    wav: floatToWavDataURL(buf, FAKE_SR),
    peaks: peaksDataURL(buf, FAKE_SR, FAKE_BPS),
    duration_s: durS,
    model: opts.model || '7-9-adr',
    checkpoint: opts.checkpoint || 'checkpoint_final.pt',
    prior_mode: opts.prior_mode || 'bend',
    scope: opts.scope || 'phrase',
    note_ids: (sel ? opts.note_ids : null),
    device: 'mock',
    warnings: notes.length ? [] : ['project has no notes'],
  };
}

function abToBase64(ab){
  const bytes = new Uint8Array(ab); let bin = ''; const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  return btoa(bin);
}

function floatToWavDataURL(f32, sr){
  const n = f32.length;
  const ab = new ArrayBuffer(44 + n * 2), dv = new DataView(ab);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); ws(8, 'WAVE');
  ws(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  ws(36, 'data'); dv.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++){ const s = Math.max(-1, Math.min(1, f32[i])); dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true); off += 2; }
  return 'data:audio/wav;base64,' + abToBase64(ab);
}

function peaksDataURL(f32, sr, bps){
  const spb = Math.max(1, Math.round(sr / bps));
  const nBins = Math.max(1, Math.ceil(f32.length / spb));
  const ab = new ArrayBuffer(16 + nBins * 2), dv = new DataView(ab);
  ws32(dv, 0, 'SPK1'); dv.setUint32(4, sr, true); dv.setUint32(8, bps, true); dv.setUint32(12, nBins, true);
  let off = 16;
  for (let b = 0; b < nBins; b++){
    let mn = 1, mx = -1; const s = b * spb, e = Math.min(f32.length, s + spb);
    for (let i = s; i < e; i++){ const v = f32[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (e <= s){ mn = 0; mx = 0; }
    dv.setInt8(off++, Math.max(-127, Math.min(127, Math.round(mn * 127))));
    dv.setInt8(off++, Math.max(-127, Math.min(127, Math.round(mx * 127))));
  }
  return 'data:application/octet-stream;base64,' + abToBase64(ab);
}
function ws32(dv, off, s){ for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); }

// ---------- import / export ----------
// Parsing a .mid client-side is out of scope for the mock, so import is a no-op that the
// UI surfaces as "not implemented in mock". Export returns a plausible fake /media path.
export async function importMidi(){
  return { state: 'not_implemented', detail: 'MIDI import needs the real backend (parsing .mid client-side is out of scope for mock mode)' };
}
export async function exportProject(id, opts = {}){
  const kind = opts.kind === 'midi' ? 'midi' : 'wav';
  const ext = kind === 'midi' ? 'mid' : 'wav';
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  return { kind, path: `data:text/plain,mock-export`, mock: true,
           filename: `${id}-${ts}.${ext}`, warnings: [] };
}
