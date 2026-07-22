// mock.js — in-browser fake backend (?mock=1) so the frontend runs with no server,
// plus the in-page selftest (?mock=1&selftest=1). Same code paths as production;
// only api.js is diverted here.

const SR = 44100, HOP = 512, DUR = 20.0;
const FMAX = 8000;   // mock mel fmax (server provides the real filterbank)

const ARTS = [
  { name: 'normal',   key: '1', color: '#5b9dff', abbrev: 'nrm'  },
  { name: 'slur',     key: '2', color: '#7ee0c0', abbrev: 'slur' },
  { name: 'spiccato', key: '3', color: '#f2b455', abbrev: 'spic' },
  { name: 'detache',  key: '4', color: '#c98bf2', abbrev: 'det'  },
];

const melHz = (m) => 440 * Math.pow(2, (m - 69) / 12);
const mel = (f) => 2595 * Math.log10(1 + f / 700);
const melBin = (m) => Math.max(0, Math.min(127, (mel(melHz(m)) / mel(FMAX)) * 127));

// ---------- documents (kept in memory across PUTs) ----------
let mockDoc = buildDoc();
let mockRev = mockDoc.rev;
let metaCache = null;

function buildDoc(){
  const seq = [55,57,59,60,62,64,66,67,69,71,72,74,76,78,79,81,83,84,86,88,86,83,79,76];
  const vibNotes = new Set([9, 13, 19]);
  const humanNotes = new Set([0, 1]);
  const notes = [];
  let t = 0.4;
  seq.forEach((pitch, i) => {
    const dur = 0.6, start = t, end = t + dur; t += 0.75;
    const vel = 60 + Math.round(30 * Math.sin(i * 0.7) + 10);
    notes.push({
      id: 'n' + String(i).padStart(4, '0'),   // 0-based, matches server note_id(ordinal)
      start_s: +start.toFixed(3), end_s: +end.toFixed(3), pitch,
      velocity: Math.max(1, Math.min(127, vel)),
      technique: ARTS[i % 4].name,
      vibrato: vibNotes.has(i),
      slur_group: null,
      auto: {
        velocity: Math.max(1, Math.min(127, vel)), technique: ARTS[i % 4].name,
        vibrato: vibNotes.has(i), amplitude: +(0.3 + 0.5 * Math.random()).toFixed(3),
        onset_strength: +(1 + Math.random()).toFixed(3),
        vibrato_rate_hz: vibNotes.has(i) ? +(5 + Math.random()).toFixed(2) : 0,
        vibrato_extent_cents: vibNotes.has(i) ? Math.round(20 + 20 * Math.random()) : 0,
      },
      human: { technique: humanNotes.has(i), vibrato: false, velocity: false, timing: false },
    });
  });
  return {
    schema_version: 1,
    clip_id: 'mock_scale',
    rev: 1,
    next_note_ordinal: seq.length,
    sr: SR, hop: HOP,
    duration_s: DUR,
    offset_applied_s: -0.031,
    audio: { sr: SR, hop: HOP },
    pipeline: {},
    vocabulary: ARTS.map((a) => ({ ...a })),
    notes,
    mute_regions: [{ start_s: 8.0, end_s: 9.0, label: 'page turn' }],
    orphaned_labels: [
      { pitch: 67, start_s: 5.12, end_s: 5.7, technique: 'spiccato', vibrato: false, velocity: 88, human: { technique: true }, reason: 'no_match' },
      { pitch: 74, start_s: 11.9, end_s: 12.4, technique: 'slur', vibrato: true, velocity: 70, human: { vibrato: true }, reason: 'no_match' },
    ],
    view: { px_per_sec: 200.0, scroll_s: 0.0, gt_variant: 'cleaned' },
  };
}

// ---------- mel tile PNG via offscreen canvas ----------
function magma(x){
  const stops = [[0,0,4],[80,18,123],[182,54,121],[251,136,97],[252,253,191]];
  x = Math.max(0, Math.min(1, x)); const s = x * (stops.length - 1);
  const i = Math.floor(s), f = s - i, a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function melTileDataURL(w, h = 128){
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d');
  const img = g.createImageData(w, h);
  const frameDur = HOP / SR;
  // base field: faint low-bin energy + noise
  for (let y = 0; y < h; y++){
    const bin = (h - 1 - y);                       // image bottom = bin 0
    for (let x = 0; x < w; x++){
      let v = 0.05 + 0.10 * Math.exp(-bin / 40) + 0.03 * Math.random();
      const [r, gg, b] = magma(v);
      const o = (y * w + x) * 4; img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  // harmonic stripes over notes
  const paint = (frame, bin, val) => {
    const y = Math.round(h - 1 - bin);
    for (let dy = -1; dy <= 1; dy++){
      const yy = y + dy; if (yy < 0 || yy >= h) continue;
      const o = (yy * w + frame) * 4;
      const [r, gg, b] = magma(val * (dy === 0 ? 1 : 0.6));
      if (r + gg + b > img.data[o] + img.data[o + 1] + img.data[o + 2]){ img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; }
    }
  };
  for (const n of mockDoc.notes){
    const f0 = Math.floor(n.start_s / frameDur), f1 = Math.min(w - 1, Math.floor(n.end_s / frameDur));
    for (let harm = 1; harm <= 5; harm++){
      const bin = melBin(n.pitch + 12 * Math.log2(harm));
      const val = 0.85 / harm;
      for (let f = f0; f <= f1 && f < w; f++) paint(f, bin, val);
    }
  }
  g.putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}

function buildMeta(){
  if (metaCache) return metaCache;
  const mbo = new Array(128); for (let m = 0; m < 128; m++) mbo[m] = melBin(m);
  const nFrames = Math.ceil(DUR * SR / HOP);   // single ragged tile (< tile_frames)
  const stretch = { source: {}, cleaned: {} };
  for (const v of ['source', 'cleaned']) for (const s of ['0.5', '0.25']) stretch[v][s] = `mock:${v}:${s}`;
  metaCache = {
    clip_id: 'mock_scale', duration_s: DUR, offset_applied_s: -0.031, sr: SR, hop: HOP, has_notes: true,
    mel: {
      n_frames: nFrames, n_mels: 128, tile_frames: 2048, n_tiles: 1,
      tiles: ['tile_0000.png'], tile_widths: [nFrames], frame_rate: SR / HOP,
      sr: SR, hop: HOP, mel_bin_of_midi: mbo,
    },
    media: {
      source: 'mock:source:1', cleaned: 'mock:cleaned:1', stretch,
      onset_env: 'mock:onset', mel_tiles: [melTileDataURL(nFrames)],
    },
  };
  return metaCache;
}

// ---------- API surface (consumed by api.js) ----------
export async function getConfig(){
  return { articulations: ARTS, default_articulation: 'normal', editing_enabled: true,
           speeds: [1, 0.5, 0.25], sr: SR, hop: HOP, dataset_root: 'Data/recorded' };
}
let mockVerified = false;
export async function getClips(){
  return [{ id: 'mock_scale', name: 'mock_scale (demo)', duration_s: DUR, state: 'ready', has_notes: true, human_edits: true, verified: mockVerified }];
}
export async function getMeta(){ return buildMeta(); }
export async function getNotes(){ return JSON.parse(JSON.stringify({ ...mockDoc, rev: mockRev, verified: mockVerified })); }
export async function putNotes(id, doc){ mockRev = (doc.rev || mockRev) + 1; mockDoc = JSON.parse(JSON.stringify(doc)); mockDoc.rev = mockRev; if (doc.verified !== undefined) mockVerified = !!doc.verified; console.log('[mock] PUT notes -> rev', mockRev, `(${(doc.notes || []).length} notes)`); return { rev: mockRev }; }
export async function setVerified(id, verified){ mockVerified = !!verified; mockRev += 1; return { status: 'ok', rev: mockRev, verified: mockVerified }; }

let compileJob = null;
export async function startCompile(){ compileJob = { start: performance.now() }; return { status: 'accepted' }; }
export async function compileStatus(){
  const root = 'Data/datasets/gt_arky';
  if (!compileJob) return { running: false, done: 0, skipped: 0, failed: 0, current: null, total: 0, root };
  const el = performance.now() - compileJob.start, total = 1500;
  if (el >= total){ compileJob = null; return { running: false, done: 1, skipped: 0, failed: 0, current: null, total: 1, root }; }
  return { running: true, done: 0, skipped: 0, failed: 0, current: 'mock_scale', total: 1, root };
}

let job = null;
export async function processClip(id){ job = { id, start: performance.now() }; return { job_id: 'mockjob', state: 'processing' }; }
const STAGES = ['ingest','denoise','transcribe','align','velocity','vibrato','media','finalize'];
export async function getStatus(){
  if (!job) return { state: 'ready', stage: null, stage_index: 0, n_stages: 8, pct: -1, message: '', error: null };
  const el = performance.now() - job.start, total = 2200;
  if (el >= total){ job = null; return { state: 'done', stage: 'finalize', stage_index: 7, n_stages: 8, pct: 100, message: 'done', error: null }; }
  const idx = Math.min(STAGES.length - 1, Math.floor(el / total * STAGES.length));
  return { state: 'processing', stage: STAGES[idx], stage_index: idx, n_stages: 8, pct: Math.round(el / total * 100), message: STAGES[idx], error: null };
}

export async function getOnsetEnv(){
  const n = Math.ceil(DUR * SR / HOP);
  const env = new Float32Array(n);
  for (let i = 0; i < n; i++) env[i] = 0.05 + 0.03 * Math.random();
  const frameDur = HOP / SR;
  for (const note of mockDoc.notes){
    const f = Math.floor(note.start_s / frameDur);
    for (let k = 0; k < 12 && f + k < n; k++) env[f + k] += 1.4 * Math.exp(-k / 3);
  }
  return env;
}

export async function decodeAudio(meta, variant, speed, ctx){
  // silent buffer of the prestretched length (duration/speed); timing is what matters.
  const len = Math.max(1, Math.ceil((DUR / speed) * ctx.sampleRate));
  return ctx.createBuffer(1, len, ctx.sampleRate);
}

// ==================================================================
// Selftest
// ==================================================================
export async function runSelfTest(d){
  const R = [];
  const ok = (cond, msg) => R.push({ ok: !!cond, msg });
  const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

  try {
    const { store, apply, undo, redo, select, notesSorted, noteById, nextNote, timeline, edit, keymap, player, actions } = d;
    const s = notesSorted();
    const a = s[0], b = s[1];

    // 1) apply / undo / redo integrity (incl. selection capture)
    select(a.id);
    const oldVel = a.velocity;
    apply(edit.setFields(a.id, { velocity: 99 }, { label: 't', selAfter: b.id, phAfter: b.start_s }));
    ok(noteById(a.id).velocity === 99, 'apply mutates field');
    ok(store.selectedId === b.id, 'apply sets selAfter');
    ok(near(store.playhead, b.start_s), 'apply sets phAfter');
    undo();
    ok(noteById(a.id).velocity === oldVel, 'undo restores field');
    ok(store.selectedId === a.id, 'undo restores selBefore');
    redo();
    ok(noteById(a.id).velocity === 99, 'redo re-applies');
    ok(store.selectedId === b.id, 'redo restores selAfter');
    undo();   // leave clean

    // 2) timeline transform round-trips
    store.view.pxPerSec = 200; store.view.originS = 3;
    timeline.layout.width = 800; timeline.layout.rollH = 360;
    ok([0, 5.5, 12.3].every((t) => near(timeline.xToTime(timeline.timeToX(t)), t, 1e-6)), 'time<->x round-trip');
    ok([0, 40, 90, 127].every((bin) => near(timeline.yToBin(timeline.binToY(bin)), bin, 1e-6)), 'bin<->y round-trip');

    // 3) keymap "1" sets technique + auto-advances selection (use a note with no
    //    prior human edits so the flag flips observably false -> true)
    const c = s[5], cNext = s[6];
    const cTechOld = c.technique, cHumanOld = c.human.technique;
    select(c.id);
    keymap.dispatch({ key: '1' });
    ok(store.selectedId === cNext.id, 'keymap 1 auto-advances');
    ok(noteById(c.id).technique === ARTS[0].name, 'keymap 1 sets technique');
    ok(noteById(c.id).human.technique === true, 'keymap 1 flags human.technique');
    undo();
    ok(noteById(c.id).human.technique === cHumanOld, 'undo of label restores human flag');
    ok(noteById(c.id).technique === cTechOld, 'undo of label restores technique');

    // 4) space toggles player state (mock buffers)
    if (player && player.ready){
      try { await player.ensureBuffer(); } catch {}
      keymap.dispatch({ key: ' ' });
      ok(player.isPlaying === true, 'space starts playback');
      keymap.dispatch({ key: ' ' });
      ok(player.isPlaying === false, 'space pauses playback');
    } else ok(true, 'player unavailable (skipped)');

    // 5) velocity mapping edit (command + clamping + human flag)
    select(a.id);
    apply(edit.setVelocity(a.id, 200));
    ok(noteById(a.id).velocity === 127, 'velocity clamps to 127');
    ok(noteById(a.id).human.velocity === true, 'velocity flags human.velocity');
    undo();
    apply(edit.setVelocity(a.id, -5));
    ok(noteById(a.id).velocity === 1, 'velocity clamps to 1');
    undo();
  } catch (e) {
    R.push({ ok: false, msg: 'exception: ' + (e && e.stack || e) });
  }

  const pass = R.filter((r) => r.ok).length, total = R.length;
  const fails = R.filter((r) => !r.ok);
  const summary = fails.length
    ? `SELFTEST FAIL ${pass}/${total}: ${fails[0].msg}`
    : `SELFTEST PASS ${pass}/${total}`;
  document.title = summary;
  const sink = document.getElementById('selftest-result');
  if (sink) sink.textContent = summary + '\n' + R.map((r) => `${r.ok ? 'ok  ' : 'FAIL'} ${r.msg}`).join('\n');
  (fails.length ? console.error : console.log)(summary);
  R.forEach((r) => (r.ok ? console.log : console.error)(`${r.ok ? 'ok' : 'FAIL'}: ${r.msg}`));
  return summary;
}
