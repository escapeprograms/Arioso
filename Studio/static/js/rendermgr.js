// rendermgr.js — render orchestration + the render UX around it. Owns the toolbar Render
// button, the transport progress strip, and the playback-source indicator. On render it
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

  const btn = $('btn-render');
  if (btn) btn.onclick = () => render();

  const src = $('src-ind');
  if (src) src.onclick = () => { player.setForceSynth(!player.forceSynth); updateSourceIndicator(); };

  subscribe(updateSourceIndicator);
  onMutation(onDocMutated);

  showProgress(false);
  updateSourceIndicator();
}

// ---------- staleness reaction ----------
function onDocMutated(){
  // state.markDirty already set renderInfo.fresh = false.
  if (player.isPlaying && player.renderActive && !player.useRender()) player.reschedule();  // -> synth
  updateSourceIndicator();
  requestStatic();   // redraw the waveform lane dimmed
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

// ---------- playback-source indicator ("MODEL" / "SYNTH") ----------
function updateSourceIndicator(){
  const el = $('src-ind'); if (!el) return;
  const usingRender = player.useRender();
  const lbl = el.querySelector('.lbl');
  el.dataset.src = usingRender ? 'model' : 'synth';
  el.classList.toggle('forced', player.forceSynth && player.hasRender);
  if (lbl) lbl.textContent = usingRender ? 'MODEL' : 'SYNTH';
  if (!player.hasRender) el.title = 'No render yet — preview synth';
  else if (player.forceSynth) el.title = 'Forced to synth — click to allow the model render';
  else if (usingRender) el.title = 'Playing the model render — click to force synth';
  else el.title = 'Render is stale — click to toggle synth lock';
}
