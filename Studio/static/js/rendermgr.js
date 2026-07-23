// rendermgr.js — render orchestration + the render UX around it. Owns the toolbar Render
// button, the transport progress strip, and the playback-source indicator. On render it
// first awaits a save flush (the server renders the on-disk project.json), then
// POSTs (scope "selection" with the current selection when non-empty, else "phrase") using
// the DEV drawer's model/checkpoint/prior_mode, polls status every 500 ms to drive the
// progress UI, and on completion loads the peaks (waveform lane) + decodes the wav (player
// rendered source), marks the render fresh, and flashes success.
//
// Freshness is flipped to false centrally in state.markDirty() (every doc mutation passes
// through it); here we register an onMutation hook to dim the waveform, refresh the source
// indicator, and — if the rendered buffer is the live playing source — fall back to synth.
import { store, subscribe, selectedIds, onMutation } from './state.js';
import * as api from './api.js';
import * as devopts from './devopts.js';
import * as waveform from './waveform.js';
import { player } from './player.js';

const $ = (id) => document.getElementById(id);

let flash = () => {};
let setStatus = () => {};
let requestStatic = () => {};
let flushSave = async () => {};

let polling = false;
let pollTimer = null;
let running = false;

// Callers that need to await a render's completion (e.g. export-WAV auto-render before
// retry) register here; every terminal path settles them with ok=true/false.
let waiters = [];
function settle(ok){ const w = waiters; waiters = []; for (const r of w) r(ok); }

// Kick a render and resolve when it finishes (true) or fails / can't start (false).
export function renderAndWait(){ return new Promise((res) => { waiters.push(res); render(); }); }
export function isRunning(){ return running; }

export function init(deps = {}){
  flash = deps.flashStatus || (() => {});
  setStatus = deps.setStatus || (() => {});
  requestStatic = deps.requestStatic || (() => {});
  flushSave = deps.flushSave || (async () => {});

  const btn = $('btn-render');
  if (btn) btn.onclick = () => render();

  const src = $('src-ind');
  if (src) src.onclick = () => { cycleSource(); };

  subscribe(updateSourceIndicator);
  onMutation(onDocMutated);

  showProgress(false);
  updateSourceIndicator();
}

// ---------- staleness reaction ----------
function onDocMutated(){
  // state.markDirty already set renderInfo.fresh = false.
  priorFresh = false;   // the cached prior no longer matches the doc; re-render on next switch
  if (player.isPlaying && player.renderActive && !player.useRender()) player.reschedule();  // -> synth
  updateSourceIndicator();
  requestStatic();   // redraw the waveform lane dimmed
}

// ---------- playback-source cycling (MODEL -> PRIOR -> SYNTH) ----------
// The prior source is torch-free and cheap, so it is rendered lazily on demand and
// re-rendered whenever the document changed since the last prior render (priorFresh).
let priorFresh = false;
let priorBusy = false;

// Ensure a fresh prior buffer is loaded; returns false (and stays off PRIOR) on failure.
async function ensurePrior(){
  if (priorFresh && player.hasPrior()) return true;
  if (!store.projectId){ flash('no project'); return false; }
  if (priorBusy) return false;
  priorBusy = true;
  setIndBusy(true);
  try {
    // The server renders the on-disk project.json — flush the debounced autosave first.
    await flushSave();
    if (store.save.state !== 'saved'){ flash('save failed — prior aborted'); return false; }
    const opts = devopts.getOpts();
    const meta = await api.renderPrior(store.projectId, { prior_mode: opts.prior_mode });
    if (meta && meta.wav) await player.loadPriorWav(meta.wav);
    else { flash('prior render produced no audio'); return false; }
    priorFresh = true;
    return true;
  } catch (err){
    surfaceError(err);
    return false;
  } finally {
    priorBusy = false;
    setIndBusy(false);
    updateSourceIndicator();
  }
}

async function cycleSource(){
  if (priorBusy) return;
  const order = ['auto', 'prior', 'synth'];
  const next = order[(order.indexOf(player.sourceMode) + 1) % order.length];
  if (next === 'prior'){
    const ok = await ensurePrior();
    if (!ok){ player.setSourceMode('synth'); updateSourceIndicator(); return; }
  }
  player.setSourceMode(next);
  updateSourceIndicator();
}

// ---------- the render action ----------
export async function render(){
  if (running){ flash('render already running'); settle(false); return; }
  if (!store.projectId){ flash('no project'); settle(false); return; }

  const ids = selectedIds();
  const scope = ids.length ? 'selection' : 'phrase';
  const opts = { scope, ...devopts.getOpts() };
  if (scope === 'selection') opts.note_ids = ids;

  running = true;
  setBtnBusy(true);
  showProgress(true);
  setProgress('serialize', 0);
  setStatus(`rendering ${scope}${scope === 'selection' ? ` (${ids.length})` : ''}…`);

  // The server renders the on-disk project.json — flush the (debounced) autosave first
  // or a render right after an edit would synthesize the previous document.
  await flushSave();
  if (store.save.state !== 'saved'){
    running = false; setBtnBusy(false); showProgress(false);
    flash('save failed — render aborted');
    settle(false);
    return;
  }

  try {
    await api.startRender(store.projectId, opts);
  } catch (err){
    running = false; setBtnBusy(false); showProgress(false);
    if (err && err.status === 409){ flash('render already running'); }
    else { surfaceError(err); }
    settle(false);
    return;
  }
  startPolling();
}

function startPolling(){
  if (polling) return;
  polling = true;
  const poll = async () => {
    if (!polling) return;
    let st;
    try { st = await api.getRenderStatus(store.projectId); }
    catch (err){ stopPolling(); running = false; setBtnBusy(false); showProgress(false); surfaceError(err); settle(false); return; }

    const state = st && st.state;
    if (state === 'processing'){
      setProgress(st.stage || '', typeof st.pct === 'number' ? st.pct : 0);
      pollTimer = setTimeout(poll, 500);
      return;
    }
    if (state === 'done'){ stopPolling(); await onRenderDone(); return; }
    if (state === 'error'){ stopPolling(); running = false; setBtnBusy(false); showProgress(false); surfaceError(st.error || 'render failed'); settle(false); return; }
    // idle / unknown — treat as finished; try to pick up meta.
    stopPolling(); await onRenderDone();
  };
  poll();
}
function stopPolling(){ polling = false; if (pollTimer){ clearTimeout(pollTimer); pollTimer = null; } }

async function onRenderDone(){
  setProgress('write', 100);
  let meta;
  try { meta = await api.getRenderMeta(store.projectId); }
  catch (err){ running = false; setBtnBusy(false); showProgress(false); surfaceError(err); return; }

  try {
    await loadRender(meta);
    store.renderInfo.meta = meta;
    store.renderInfo.fresh = true;   // set directly (not via markDirty) so it stays fresh
    if (player.isPlaying) player.reschedule();   // switch the live pass to the rendered buffer
    devopts.setReadout(meta);
    const warns = (meta.warnings && meta.warnings.length) ? ` (${meta.warnings.join('; ')})` : '';
    flash('render complete' + warns);
  } catch (err){
    surfaceError(err);
  } finally {
    running = false; setBtnBusy(false);
    showProgress(false);
    updateSourceIndicator();
    requestStatic();
    settle(true);
  }
}

// fetch peaks + decode wav. wav may be null (backend chose to skip) -> stay on synth.
async function loadRender(meta){
  if (meta.peaks) await waveform.load(meta.peaks); else waveform.clear();
  if (meta.wav) await player.loadRenderWav(meta.wav); else player.clearRenderBuffer();
}

// ---------- boot / project-load: adopt any existing (but unverifiable, thus stale) render ----------
export async function onProjectLoaded(){
  stopPolling();
  running = false; setBtnBusy(false); showProgress(false);
  player.clearRenderBuffer();
  player.clearPriorBuffer();
  player.setSourceMode('auto');
  priorFresh = false;
  waveform.clear();
  store.renderInfo.fresh = false;
  store.renderInfo.meta = null;
  requestStatic();
  updateSourceIndicator();
  try {
    const meta = await api.getRenderMeta(store.projectId);
    if (meta && (meta.peaks || meta.wav)){
      await loadRender(meta);
      store.renderInfo.meta = meta;
      // Left stale: we cannot cheaply prove it matches the current doc, so playback stays on
      // synth and the lane shows the dimmed "re-render" hint until the user renders again.
      devopts.setReadout(meta);
      requestStatic();
      updateSourceIndicator();
    }
  } catch { /* 404 not_rendered — leave the empty lane hint */ }
}

// ---------- progress UI ----------
function showProgress(on){ const el = $('render-progress'); if (el) el.hidden = !on; }
function setProgress(stage, pct){
  const fill = $('rp-fill'), lbl = $('rp-stage');
  if (fill) fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
  if (lbl) lbl.textContent = stage || '';
}
function setBtnBusy(on){ const b = $('btn-render'); if (b){ b.classList.toggle('busy', on); b.disabled = on; } }

function surfaceError(err){
  const msg = (err && (err.detail || err.message)) || String(err) || 'render error';
  setStatus('render error: ' + msg, 'error');
  const b = $('btn-render'); if (b) b.title = 'Render failed: ' + msg;
}

// ---------- playback-source indicator ("MODEL" / "PRIOR" / "SYNTH") ----------
// Busy state shown while the prior renders; the indicator otherwise reflects the
// active source-cycle mode (click cycles MODEL -> PRIOR -> SYNTH).
function setIndBusy(on){
  const el = $('src-ind'); if (!el) return;
  el.classList.toggle('busy', on);
  if (on){
    el.dataset.src = 'prior';
    const lbl = el.querySelector('.lbl'); if (lbl) lbl.textContent = 'PRIOR';
    el.title = 'Rendering prior…';
  }
}

function updateSourceIndicator(){
  const el = $('src-ind'); if (!el) return;
  if (priorBusy) return;                       // setIndBusy owns the display while rendering
  const lbl = el.querySelector('.lbl');
  const mode = player.sourceMode;
  el.classList.remove('forced', 'busy');
  if (mode === 'prior'){
    el.dataset.src = 'prior';
    if (lbl) lbl.textContent = 'PRIOR';
    el.title = 'Playing the raw saw prior — click for synth';
  } else if (mode === 'synth'){
    el.dataset.src = 'synth';
    el.classList.toggle('forced', player.hasRender);
    if (lbl) lbl.textContent = 'SYNTH';
    el.title = 'Preview synth — click to cycle back to the model render';
  } else {                                      // 'auto'
    const usingRender = player.useRender();
    el.dataset.src = usingRender ? 'model' : 'synth';
    if (lbl) lbl.textContent = usingRender ? 'MODEL' : 'SYNTH';
    if (!player.hasRender) el.title = 'No render yet — click to audition the prior';
    else if (usingRender) el.title = 'Playing the model render — click to audition the prior';
    else el.title = 'Render is stale — click to audition the prior';
  }
}
