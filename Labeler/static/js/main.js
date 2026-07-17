// main.js — bootstrap + orchestration: loads config/clips, wires DOM (sidebar,
// transport, inspector, modal, orphans), builds keymap actions, drives the rAF loop,
// autosave, processing polling, and (with ?selftest=1) the in-page selftest.
import * as api from './api.js';
import {
  store, subscribe, notify, apply, undo, redo, select, deselect, setPlayhead,
  setDirtyHandler, clearHistory, notesSorted, noteById, nextNote, prevNote, firstNote,
  initIdCounter, nextOrdinal, pitchName, fmtTime, clamp, invalidateSorted,
} from './state.js';
import { clampView, durationS, visibleTimeRange } from './timeline.js';
import * as timeline from './timeline.js';
import * as render from './render.js';
import * as interact from './interact.js';
import * as keymap from './keymap.js';
import { player } from './player.js';
import * as edit from './editing.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const editing = () => store.config && store.config.editing_enabled;

let staticDirty = true;
const requestStatic = () => { staticDirty = true; };

// ---------------- boot ----------------
boot().catch((err) => { recordError('boot: ' + (err && err.stack || err)); });

async function boot(){
  installErrorTrap();
  store.config = await api.getConfig();
  store.clips = await api.getClips();

  render.init(requestStatic);           // tile load -> redraw
  interact.init({ requestStatic });
  keymap.init(actions);
  player.init((v, s) => api.decodeAudio(store.meta, v, s, player.ctx));
  setDirtyHandler(scheduleSave);

  buildTechDropdown();
  wireTransport();
  wireModal();
  wireOrphan();
  subscribe(onStoreChange);
  window.addEventListener('keydown', keymap.onKeyDown);
  window.addEventListener('resize', onResize);
  new ResizeObserver(onResize).observe($('roll'));
  window.addEventListener('beforeunload', flushBeacon);

  renderSidebar();
  onResize();
  startRenderLoop();

  const first = store.clips.find((c) => c.state === 'ready') || store.clips[0];
  if (first) await loadClip(first.id);
  onResize();
  syncTransport();

  if (api.SELFTEST) runSelfTest();
}

// ---------------- clip loading ----------------
async function loadClip(id){
  store.clipId = id;
  store.meta = await api.getMeta(id);
  render.resetColors();
  store.doc = await api.getNotes(id);
  if (!store.doc.notes) store.doc.notes = [];
  if (!store.doc.mute_regions) store.doc.mute_regions = [];
  invalidateSorted();  // renders before this fetch resolved cached an empty list
  initIdCounter(store.doc);
  store.onset = await api.getOnsetEnv(store.meta);
  clearHistory();
  restoreView(store.doc.view);
  store.selectedId = (firstNote() && firstNote().id) || null;
  store.playhead = 0;
  store.save.state = 'saved';
  player.reset();
  onResize();
  renderSidebar();
  renderOrphans();
  requestStatic();
  onStoreChange();
  // warm audio buffers in the background
  try { player.prefetch(); } catch {}
}

// notes.json view schema is snake_case (px_per_sec/scroll_s/gt_variant).
function restoreView(dv){
  if (!dv) return;
  if (dv.px_per_sec) store.view = { pxPerSec: dv.px_per_sec, originS: dv.scroll_s || 0 };
  else if (dv.pxPerSec) store.view = { pxPerSec: dv.pxPerSec, originS: dv.originS || 0 };
  if (dv.gt_variant === 'source' || dv.gt_variant === 'cleaned') store.variant = dv.gt_variant;
}
function foldViewIntoDoc(){
  if (!store.doc) return;
  store.doc.view = { px_per_sec: store.view.pxPerSec, scroll_s: store.view.originS, gt_variant: store.variant };
  store.doc.next_note_ordinal = Math.max(store.doc.next_note_ordinal | 0, nextOrdinal());
}

// ---------------- actions (keymap + inspector) ----------------
const actions = {
  setTechniqueByIndex(i){
    const arts = store.config.articulations;
    if (i >= arts.length || !store.selectedId) return;
    const sel = store.selectedId, nx = nextNote(sel);
    apply(edit.setTechnique(sel, arts[i].name,
      { selAfter: nx ? nx.id : sel, phAfter: nx ? nx.start_s : store.playhead }));
  },
  toggleVibrato(){ if (store.selectedId) apply(edit.toggleVibrato(store.selectedId)); },
  selectPrev(){ const p = store.selectedId ? prevNote(store.selectedId) : firstNote(); if (p) select(p.id, { playhead: p.start_s }); },
  selectNext(){ const n = store.selectedId ? nextNote(store.selectedId) : firstNote(); if (n) select(n.id, { playhead: n.start_s }); },
  velUp(){ if (store.selectedId){ const n = noteById(store.selectedId); apply(edit.setVelocity(store.selectedId, n.velocity + 5)); } },
  velDown(){ if (store.selectedId){ const n = noteById(store.selectedId); apply(edit.setVelocity(store.selectedId, n.velocity - 5)); } },
  undo(){ undo(); }, redo(){ redo(); },
  speedDown(){ changeSpeed(-1); }, speedUp(){ changeSpeed(+1); },
  toggleMuteOrig(){ player.toggleMute('orig'); syncTransport(); },
  toggleMuteMidi(){ player.toggleMute('midi'); syncTransport(); },
  toggleFollow(){ store.follow = !store.follow; },
  save(){ saveNow(); },
  playPause(){ playPause(); },
  deleteSelected(){ if (editing() && store.selectedId) apply(edit.deleteNote(store.selectedId)); },
  splitAtPlayhead(){ if (editing() && store.selectedId){ const c = edit.splitNote(store.selectedId, store.playhead); if (c) apply(c); } },
  togglePaintRegion(){ store.paintMode = !store.paintMode; $('region-popover').hidden = !store.paintMode; if (store.paintMode) renderRegionList(); requestStatic(); },
  deselect(){ deselect(); },
};

function changeSpeed(delta){   // delta +1 => faster
  const sp = (store.config.speeds || [1, 0.5, 0.25]).slice().sort((a, b) => b - a);
  let i = sp.indexOf(store.speed); if (i < 0) i = 0;
  i = clamp(i - delta, 0, sp.length - 1);
  player.setSpeed(sp[i]); syncTransport(); syncPlayBtn();
}

function playPause(){
  if (player.isPlaying){
    player.pause();
    const sel = store.selectedId ? noteById(store.selectedId) : null;
    store.playhead = sel ? sel.start_s : player.startClipTime;
    notify();
  } else {
    player.play(store.playhead);
  }
  syncPlayBtn();
}

// ---------------- render loop ----------------
function startRenderLoop(){
  const loop = () => { try { tick(); } catch (e) { recordError('tick: ' + (e && e.stack || e)); } requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
}
let lastPPS = -1;
function tick(){
  if (player.isPlaying){
    store.playhead = player.clipTime();
    handleFollow();
    if (store.playhead >= durationS() - 1e-3){ player.pause(); syncPlayBtn(); }
  }
  if (staticDirty){ render.drawStage(); staticDirty = false; }
  render.drawOverlay();
  $('decoding-chip').hidden = !store.decoding;
  if (store.view.pxPerSec !== lastPPS){ lastPPS = store.view.pxPerSec; $('zoom-ind').textContent = Math.round(lastPPS) + ' px/s'; }
}
function handleFollow(){
  if (!store.follow) return;
  const [t0, t1] = visibleTimeRange();
  const w = t1 - t0;
  if (store.playhead > t0 + w * 0.85 || store.playhead < t0){
    store.view.originS = store.playhead - w * 0.5; clampView(); staticDirty = true;
  }
}

// ---------------- autosave ----------------
let saveTimer = null, retryDelay = 1000;
function scheduleSave(){ updateSaveInd(); clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, 1500); }
function scheduleRetry(){ clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, retryDelay); retryDelay = Math.min(retryDelay * 2, 15000); }

async function saveNow(){
  if (!store.doc || store.clipId == null) return;
  if (store.save.state === 'saved') return;
  clearTimeout(saveTimer);
  store.save.state = 'saving'; updateSaveInd();
  foldViewIntoDoc();
  try {
    const res = await api.putNotes(store.clipId, store.doc);
    if (res && res.rev != null) store.doc.rev = res.rev;
    store.save.state = 'saved'; store.save.err = null; retryDelay = 1000;
  } catch (err) {
    if (err instanceof api.ApiError && err.status === 409){
      if (err.code === 'rev_conflict'){ store.save.state = 'error'; showRevModal(); }
      else { store.save.state = 'unsaved'; scheduleRetry(); }   // job_running etc.
    } else { store.save.state = 'error'; store.save.err = String(err); scheduleRetry(); }
  }
  updateSaveInd();
}

function flushBeacon(){
  if (store.doc && store.clipId != null && store.save.state !== 'saved'){
    foldViewIntoDoc();
    api.beaconNotes(store.clipId, store.doc);
  }
}

// ---------------- UI wiring ----------------
function buildTechDropdown(){
  const sel = $('insp-tech'); sel.innerHTML = '';
  for (const a of store.config.articulations){
    const o = document.createElement('option'); o.value = a.name; o.textContent = `${a.abbrev || a.name} (${a.key})`;
    sel.appendChild(o);
  }
}

function wireTransport(){
  $('btn-play').onclick = () => playPause();
  $('speed-seg').querySelectorAll('button').forEach((b) => {
    b.onclick = () => { player.setSpeed(parseFloat(b.dataset.speed)); syncTransport(); syncPlayBtn(); };
  });
  $('variant-toggle').querySelectorAll('button').forEach((b) => {
    b.onclick = () => { player.setVariant(b.dataset.variant); syncTransport(); syncPlayBtn(); };
  });
  $('orig-vol').oninput = (e) => player.setVolume('orig', parseFloat(e.target.value));
  $('midi-vol').oninput = (e) => player.setVolume('midi', parseFloat(e.target.value));
  $('orig-mute').onclick = () => { player.toggleMute('orig'); syncTransport(); };
  $('midi-mute').onclick = () => { player.toggleMute('midi'); syncTransport(); };

  $('insp-vel').onchange = (e) => { if (store.selectedId) apply(edit.setVelocity(store.selectedId, parseInt(e.target.value, 10))); };
  $('insp-tech').onchange = (e) => { if (store.selectedId) apply(edit.setTechnique(store.selectedId, e.target.value)); };
  $('insp-vib').onchange = (e) => { if (store.selectedId) apply(edit.setFields(store.selectedId, { vibrato: e.target.checked, 'human.vibrato': true }, { label: 'vibrato' })); };
  $('region-pop-close').onclick = () => { store.paintMode = false; $('region-popover').hidden = true; };
}

function wireModal(){
  $('rev-reload').onclick = async () => {
    $('rev-modal').hidden = true;
    store.doc = await api.getNotes(store.clipId);
    invalidateSorted();
    initIdCounter(store.doc); clearHistory();
    store.save.state = 'saved'; requestStatic(); onStoreChange();
  };
  $('rev-keep').onclick = () => { $('rev-modal').hidden = true; store.save.state = 'unsaved'; updateSaveInd(); };
}
function showRevModal(){ $('rev-modal').hidden = false; }

function wireOrphan(){
  $('orphan-head').onclick = () => {
    const b = $('orphan-body'); b.hidden = !b.hidden;
    $('orphan-head').classList.toggle('open', !b.hidden);
  };
}
function renderOrphans(){
  const orphans = (store.doc && store.doc.orphaned_labels) || [];
  $('orphan-panel').hidden = orphans.length === 0;
  $('orphan-count').textContent = orphans.length;
  const body = $('orphan-body'); body.innerHTML = '';
  for (const o of orphans){
    const d = document.createElement('div'); d.className = 'orphan-item';
    const p = o.pitch != null ? pitchName(o.pitch) : '?';
    d.textContent = `${p}  ${o.start_s != null ? o.start_s.toFixed(3) + 's' : ''}  ${o.technique || ''}${o.vibrato ? ' vib' : ''}`;
    body.appendChild(d);
  }
}

function renderRegionList(){
  const ul = $('region-list'); ul.innerHTML = '';
  const regs = (store.doc && store.doc.mute_regions) || [];
  regs.forEach((r, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${r.start_s.toFixed(2)}–${r.end_s.toFixed(2)}s</span><button>del</button>`;
    li.querySelector('button').onclick = () => { apply(edit.deleteMuteRegion(i)); renderRegionList(); };
    ul.appendChild(li);
  });
}

// ---------------- sidebar / processing ----------------
function renderSidebar(){
  const ul = $('clip-list'); ul.innerHTML = '';
  for (const c of store.clips){
    const li = document.createElement('li');
    li.className = 'clip-item' + (c.id === store.clipId ? ' active' : '');
    li.innerHTML =
      `<div class="clip-row"><span class="clip-name">${esc(c.name || c.id)}</span><span class="badge ${c.state}">${c.state}</span></div>` +
      `<div class="clip-sub"><span>${c.duration_s ? c.duration_s.toFixed(1) + 's' : ''}</span>` +
      `${c.human_edits ? '<span class="clip-edits">&#9998; edits</span>' : ''}` +
      `<span class="spacer" style="flex:1"></span>` +
      `<button class="pbtn">${c.state === 'raw' ? 'process' : 'reprocess'}</button></div>`;
    li.querySelector('.clip-row').onclick = () => loadClip(c.id);
    li.querySelector('.pbtn').onclick = (e) => { e.stopPropagation(); startProcess(c.id); };
    ul.appendChild(li);
  }
}

async function startProcess(id){
  try { await api.processClip(id, { notes_mode: 'keep_labels' }); }
  catch (err) { recordError('process: ' + err); return; }
  const clip = store.clips.find((c) => c.id === id); if (clip) clip.state = 'processing';
  renderSidebar();
  pollStatus(id);
}
function pollStatus(id){
  $('sidebar-progress').hidden = false;
  const timer = setInterval(async () => {
    let st; try { st = await api.getStatus(id); } catch { clearInterval(timer); $('sidebar-progress').hidden = true; return; }
    store.processing = st;
    const nst = st.n_stages || 8;
    $('pmsg').textContent = st.message || (st.stage ? `${st.stage} (${st.stage_index || 0}/${nst})` : 'processing…');
    const fill = $('pfill');
    if (st.pct == null || st.pct < 0){ fill.classList.add('indeterminate'); fill.style.width = ''; }
    else { fill.classList.remove('indeterminate'); fill.style.width = Math.round(st.pct) + '%'; }
    if (st.state === 'ready' || st.state === 'done' || st.state === 'error'){
      clearInterval(timer); store.processing = null; $('sidebar-progress').hidden = true;
      try { store.clips = await api.getClips(); } catch {}
      renderSidebar();
      if (id === store.clipId) await loadClip(id);
    }
  }, 500);
}

// ---------------- store-driven UI refresh ----------------
function onStoreChange(){
  staticDirty = true;
  updateInspector();
  updateSaveInd();
  updateSidebarActive();
  syncTransport();
  if (store.paintMode) renderRegionList();   // keep the mute-region popover in sync
}

function updateInspector(){
  const insp = $('inspector');
  const n = store.selectedId ? noteById(store.selectedId) : null;
  if (!n){ insp.hidden = true; return; }
  insp.hidden = false;
  $('insp-pitch').textContent = pitchName(n.pitch);
  $('insp-start').textContent = n.start_s.toFixed(3);
  $('insp-end').textContent = n.end_s.toFixed(3);
  const av = document.activeElement;
  if (av !== $('insp-vel')) $('insp-vel').value = n.velocity;
  if (av !== $('insp-tech')) $('insp-tech').value = n.technique;
  if (av !== $('insp-vib')) $('insp-vib').checked = !!n.vibrato;
}

function updateSaveInd(){
  const el = $('save-ind'); el.dataset.state = store.save.state;
  const txt = { saved: 'saved ✓', saving: 'saving…', unsaved: 'unsaved •', error: 'error ⚠' }[store.save.state];
  el.querySelector('.txt').textContent = txt || store.save.state;
}
function updateSidebarActive(){
  document.querySelectorAll('.clip-item').forEach((li, i) => {
    li.classList.toggle('active', store.clips[i] && store.clips[i].id === store.clipId);
  });
}

function syncTransport(){
  $('speed-seg').querySelectorAll('button').forEach((b) => b.classList.toggle('on', parseFloat(b.dataset.speed) === store.speed));
  $('variant-toggle').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.variant === store.variant));
  $('orig-mute').classList.toggle('muted', store.tracks.orig.mute);
  $('midi-mute').classList.toggle('muted', store.tracks.midi.mute);
  const av = document.activeElement;
  if (av !== $('orig-vol')) $('orig-vol').value = store.tracks.orig.vol;
  if (av !== $('midi-vol')) $('midi-vol').value = store.tracks.midi.vol;
}
function syncPlayBtn(){
  const b = $('btn-play'); b.classList.toggle('playing', player.isPlaying);
  b.querySelector('.glyph').textContent = player.isPlaying ? '❚❚' : '▶';
}

function onResize(){ render.resize(); staticDirty = true; }

// ---------------- error trap + selftest sink ----------------
window.__errors = [];
function recordError(msg){
  window.__errors.push(msg);
  const sink = $('selftest-result'); if (sink) sink.textContent += '\nERR: ' + msg;
  if (!/^SELFTEST/.test(document.title)) document.title = 'SELFTEST ERROR: ' + msg;
  console.error(msg);
}
function installErrorTrap(){
  window.addEventListener('error', (e) => recordError('window.error: ' + (e.message || e.error)));
  window.addEventListener('unhandledrejection', (e) => recordError('unhandledrejection: ' + (e.reason && e.reason.stack || e.reason)));
}

async function runSelfTest(){
  try {
    const mock = await import('./mock.js');
    await mock.runSelfTest({
      store, apply, undo, redo, select, notesSorted, noteById, nextNote,
      timeline, edit, keymap, player, actions,
    });
  } catch (e) { recordError('selftest: ' + (e && e.stack || e)); }
}
