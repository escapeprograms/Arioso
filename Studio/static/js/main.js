// main.js — bootstrap + orchestration: loads config/models/project (creating a default if
// none), wires every module, runs the rAF draw loop with follow-playhead, autosaves via
// putProject (1.5 s debounce), flushes on unload, and drives the status footer.
import * as api from './api.js';
import {
  store, subscribe, apply, undo, redo, clearHistory, initIdCounter,
  selectAll, clearSelection, selectedIds, invalidateSorted, notesSorted, setDirtyHandler,
} from './state.js';
import { layout, clampView, visibleBeatRange, durationBeats } from './timeline.js';
import * as render from './render.js';
import * as interact from './interact.js';
import * as tools from './tools.js';
import * as keymap from './keymap.js';
import * as transport from './transport.js';
import * as clipboard from './clipboard.js';
import * as edit from './editing.js';
import * as technique from './technique.js';
import * as vellane from './vellane.js';
import * as vibrato from './vibrato.js';
import * as devopts from './devopts.js';
import * as rendermgr from './rendermgr.js';
import * as io from './io.js';
import { player } from './player.js';

const $ = (id) => document.getElementById(id);
let staticDirty = true;
const requestStatic = () => { staticDirty = true; };

boot().catch(err => setStatus('boot error: ' + (err && err.message || err), 'error'));

async function boot(){
  store.config = await api.getConfig();
  store.models = await api.getModels().catch(() => []);

  render.init(requestStatic);
  interact.init({ requestStatic });
  tools.init({ requestStatic });
  vellane.init({ requestStatic });
  technique.init({ flashStatus });
  vibrato.init();
  transport.init({ requestStatic });
  keymap.init(actions);
  player.init();
  devopts.init();
  rendermgr.init({ flashStatus, setStatus, requestStatic });
  io.init({ flashStatus, setStatus, reloadProject });
  setDirtyHandler(scheduleSave);

  subscribe(onStoreChange);
  window.addEventListener('keydown', keymap.onKeyDown);
  window.addEventListener('resize', onResize);
  new ResizeObserver(onResize).observe($('grid'));
  window.addEventListener('beforeunload', flushBeacon);
  wireLaneResize();

  await loadProject();
  onResize();
  startRenderLoop();
  transport.sync();
  setStatus('ready — mock mode: ' + (api.MOCK ? 'on' : 'off'));
}

// ---------- project loading ----------
async function loadProject(){
  const list = await api.listProjects().catch(() => []);
  // /api/projects summaries use `id` (project_store.summary); tolerate `project_id` too.
  let id = (list[0] && (list[0].id || list[0].project_id)) || null;
  if (!id){ const p = await api.createProject({ name: 'My Song' }); id = p.project_id; }
  store.projectId = id;
  applyLoadedDoc(await api.getProject(id));
}

// Re-fetch the current project and reset the editor to it (used after a MIDI import
// rewrites the notes on the server). Same reset path loadProject uses.
async function reloadProject(){
  if (store.projectId == null) return;
  applyLoadedDoc(await api.getProject(store.projectId));
}

function applyLoadedDoc(doc){
  store.doc = doc;
  if (!store.doc.notes) store.doc.notes = [];
  invalidateSorted();
  initIdCounter(store.doc);
  clearHistory();
  restoreView(store.doc.view);
  clearSelection();
  store.playhead = 0;
  store.save.state = 'saved';
  onResize();
  requestStatic();
  onStoreChange();
  rendermgr.onProjectLoaded();   // reset render state + adopt any existing (stale) render
}

function restoreView(v){
  if (!v) return;
  if (v.px_per_beat) store.view.px_per_beat = v.px_per_beat;
  if (v.scroll_beat != null) store.view.scroll_beat = v.scroll_beat;
  if (v.top_pitch != null) store.view.top_pitch = v.top_pitch;
  if (v.snap) store.snap = v.snap;
  if (v.lane) store.lane = v.lane;
  clampView();
}
function foldViewIntoDoc(){
  if (!store.doc) return;
  store.doc.view = {
    px_per_beat: store.view.px_per_beat, scroll_beat: store.view.scroll_beat,
    top_pitch: store.view.top_pitch, lane: store.lane, snap: store.snap,
  };
  store.doc.next_note_ordinal = Math.max(store.doc.next_note_ordinal | 0, notesSorted().length);
}

// ---------- keymap actions ----------
const actions = {
  undo(){ undo(); }, redo(){ redo(); },
  copy(){ clipboard.copy(); flashStatus('copied ' + selectedIds().length); },
  cut(){ clipboard.cut(); },
  paste(){ clipboard.paste(store.playhead); },
  selectAll(){ selectAll(); },
  deselect(){ clearSelection(); },
  deleteSelected(){ const ids = selectedIds(); if (ids.length) apply(edit.deleteNotes(ids)); },
  save(){ saveNow(); },
  render(){ rendermgr.render(); },
  playPause(){ transport.playPause(); },
  setTool(t){ tools.setTool(t); },
  setTechnique(digit){ technique.applyByKey(digit); },
  toggleLoop(){ transport.toggleLoop(); },
  toggleFollow(){ transport.toggleFollow(); },
};

// ---------- render loop ----------
function startRenderLoop(){
  const loop = () => { try { tick(); } catch (e) { setStatus('tick error: ' + (e && e.message || e), 'error'); } requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
}
let lastPxb = -1;
function tick(){
  if (player.isPlaying){
    store.playhead = player.clipBeat();
    followPlayhead();
    if (!store.loop.enabled && store.playhead >= durationBeats() - 1e-3){ player.pause(); }
  }
  if (staticDirty){ render.drawStage(); staticDirty = false; }
  render.drawOverlay();
  transport.sync();
  if (store.view.px_per_beat !== lastPxb){ lastPxb = store.view.px_per_beat; $('zoom-ind').textContent = Math.round(lastPxb) + ' px/beat'; }
}
function followPlayhead(){
  if (!store.follow) return;
  const [b0, b1] = visibleBeatRange();
  const w = b1 - b0;
  if (store.playhead > b0 + w * 0.85 || store.playhead < b0){
    store.view.scroll_beat = store.playhead - w * 0.5; clampView(); staticDirty = true;
  }
}

// ---------- autosave ----------
let saveTimer = null, retryDelay = 1000;
function scheduleSave(){ updateSaveInd(); clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, 1500); }
function scheduleRetry(){ clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, retryDelay); retryDelay = Math.min(retryDelay * 2, 15000); }

async function saveNow(){
  if (!store.doc || store.projectId == null) return;
  if (store.save.state === 'saved') return;
  clearTimeout(saveTimer);
  store.save.state = 'saving'; updateSaveInd();
  foldViewIntoDoc();
  try {
    const res = await api.putProject(store.projectId, store.doc);
    if (res && res.rev != null) store.doc.rev = res.rev;
    store.save.state = 'saved'; store.save.err = null; retryDelay = 1000;
  } catch (err) {
    if ((err instanceof api.ApiError && err.status === 409) || err.status === 409){
      // reload the server copy (Phase 1: last-write-wins recovery)
      try {
        store.doc = await api.getProject(store.projectId);
        invalidateSorted(); initIdCounter(store.doc); clearHistory();
        store.save.state = 'saved'; requestStatic(); onStoreChange();
        flashStatus('reloaded (save conflict)');
      } catch { store.save.state = 'error'; store.save.err = String(err); scheduleRetry(); }
    } else { store.save.state = 'error'; store.save.err = String(err); scheduleRetry(); }
  }
  updateSaveInd();
}
function flushBeacon(){
  if (store.doc && store.projectId != null && store.save.state !== 'saved'){
    foldViewIntoDoc(); api.beaconProject(store.projectId, store.doc);
  }
}

// ---------- lane resize handle ----------
function wireLaneResize(){
  const h = $('lane-resize'); if (!h) return;
  let dragging = false, sy = 0, s0 = 0;
  h.addEventListener('pointerdown', (e) => {
    dragging = true; sy = e.clientY;
    s0 = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--laneH')) || layout.laneH;
    h.setPointerCapture(e.pointerId); e.preventDefault();
  });
  h.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const nv = Math.max(40, Math.min(320, s0 + (sy - e.clientY)));
    document.documentElement.style.setProperty('--laneH', nv + 'px');
    onResize();
  });
  h.addEventListener('pointerup', (e) => { dragging = false; try { h.releasePointerCapture(e.pointerId); } catch {} });
}

// ---------- store-driven UI ----------
function onStoreChange(){ staticDirty = true; updateSaveInd(); }
function onResize(){ render.resize(); clampView(); staticDirty = true; }

function updateSaveInd(){
  const el = $('save-ind'); if (!el) return;
  el.dataset.state = store.save.state;
  el.querySelector('.txt').textContent =
    ({ saved: 'saved', saving: 'saving…', unsaved: 'unsaved', error: 'error' }[store.save.state]) || store.save.state;
}

let statusTimer = null;
function setStatus(msg, kind){ const el = $('status-msg'); if (el){ el.textContent = msg; el.dataset.kind = kind || ''; } }
function flashStatus(msg){ setStatus(msg); clearTimeout(statusTimer); statusTimer = setTimeout(() => setStatus('ready'), 1600); }
