// io.js — MIDI import + WAV/MIDI export UI. Owns the toolbar File group (Import / WAV /
// MIDI), the import-options modal (replace vs merge, adopt-BPM, snap-on-import), and the
// status-bar download link. Import uploads the .mid as raw bytes (see api.importMidi) and,
// on success, reloads the document into the store via the reloadProject callback (same
// reset main.js does on project load). Export WAV auto-renders first on a 409 render_stale
// and retries once (better UX than making the user click Render manually).
import * as api from './api.js';
import { store } from './state.js';
import * as rendermgr from './rendermgr.js';

const $ = (id) => document.getElementById(id);

let flash = () => {};
let setStatus = () => {};
let reloadProject = async () => {};

let pendingFile = null;
let importMode = 'replace';
let busy = false;

export function init(deps = {}){
  flash = deps.flashStatus || (() => {});
  setStatus = deps.setStatus || (() => {});
  reloadProject = deps.reloadProject || (async () => {});

  const fileInput = $('import-file');
  const btnImport = $('btn-import');
  if (btnImport && fileInput){
    btnImport.onclick = () => { fileInput.value = ''; fileInput.click(); };
    fileInput.onchange = () => { if (fileInput.files && fileInput.files[0]) openModal(fileInput.files[0]); };
  }

  const wav = $('btn-export-wav'), midi = $('btn-export-midi');
  if (wav) wav.onclick = () => doExport('wav');
  if (midi) midi.onclick = () => doExport('midi');

  wireModal();
}

// ---------- import modal ----------
function wireModal(){
  document.querySelectorAll('#import-mode [data-mode]').forEach(b => {
    b.onclick = () => { importMode = b.dataset.mode; syncMode(); };
  });
  const go = $('import-go'); if (go) go.onclick = runImport;
  const cancel = $('import-cancel'), x = $('import-cancel-x'), scrim = $('import-modal');
  if (cancel) cancel.onclick = closeModal;
  if (x) x.onclick = closeModal;
  if (scrim) scrim.onclick = (e) => { if (e.target === scrim) closeModal(); };
}
function syncMode(){
  document.querySelectorAll('#import-mode [data-mode]').forEach(b => b.classList.toggle('on', b.dataset.mode === importMode));
}
function openModal(file){
  pendingFile = file;
  importMode = 'replace'; syncMode();
  const fn = $('import-filename'); if (fn) fn.textContent = file.name || 'selected.mid';
  const adopt = $('import-adopt-bpm'); if (adopt) adopt.checked = false;
  const snap = $('import-snap'); if (snap) snap.checked = false;
  const m = $('import-modal'); if (m) m.hidden = false;
}
function closeModal(){
  const m = $('import-modal'); if (m) m.hidden = true;
  pendingFile = null;
  const fi = $('import-file'); if (fi) fi.value = '';
}

async function runImport(){
  if (busy || !pendingFile) return;
  if (!store.projectId){ flash('no project'); return; }
  const opts = { mode: importMode };
  if ($('import-adopt-bpm') && $('import-adopt-bpm').checked) opts.adopt_bpm = true;
  if ($('import-snap') && $('import-snap').checked) opts.snap = store.snap || 'step';

  const file = pendingFile;
  busy = true; setBusy(true);
  setStatus('importing ' + (file.name || 'MIDI') + '…');
  try {
    const res = await api.importMidi(store.projectId, file, opts);
    if (res && res.state === 'not_implemented'){
      flash(res.detail || 'import not available in mock mode');
    } else {
      await reloadProject();
      const w = (res.warnings && res.warnings.length) ? ' — ' + res.warnings.join('; ') : '';
      flash(`imported ${res.notes_added} note${res.notes_added === 1 ? '' : 's'}${w}`);
    }
  } catch (err){
    surfaceError('import', err);
  } finally {
    busy = false; setBusy(false); closeModal();
  }
}

// ---------- export ----------
async function doExport(kind){
  if (busy) return;
  if (!store.projectId){ flash('no project'); return; }
  busy = true; setBusy(true);
  setStatus(`exporting ${kind.toUpperCase()}…`);
  try {
    const res = await api.exportProject(store.projectId, { kind });
    showExportLink(res);
  } catch (err){
    if (kind === 'wav' && err && err.status === 409 && err.code === 'render_stale'){
      await exportWavAfterRender();
    } else {
      surfaceError('export', err);
    }
  } finally {
    busy = false; setBusy(false);
  }
}

// WAV export hit a stale render: render first (awaiting completion), then retry once.
async function exportWavAfterRender(){
  setStatus('render out of date — rendering before export…');
  const ok = await rendermgr.renderAndWait();
  if (!ok){ flash('render failed — cannot export WAV'); return; }
  try {
    const res = await api.exportProject(store.projectId, { kind: 'wav' });
    showExportLink(res);
  } catch (err){
    if (err && err.status === 409) flash('render still out of date — try Render, then export');
    else surfaceError('export', err);
  }
}

function showExportLink(res){
  const link = $('export-link');
  const name = (res && res.filename) || basename(res && res.path) || 'download';
  if (link && res && res.path){
    link.href = res.path;
    link.download = name;
    link.textContent = 'Download ' + name;
    link.hidden = false;
  }
  const warns = (res && res.warnings && res.warnings.length) ? ' (' + res.warnings.join('; ') + ')' : '';
  const mock = res && res.mock ? ' [mock]' : '';
  flash(`exported ${res && res.kind ? res.kind.toUpperCase() : ''}${mock}${warns}`);
}

function basename(p){ return p ? String(p).split('/').pop().split('?')[0] : ''; }

function setBusy(on){
  ['btn-import', 'btn-export-wav', 'btn-export-midi'].forEach(id => { const b = $(id); if (b) b.disabled = on; });
}

function surfaceError(what, err){
  const msg = (err && (err.detail || err.message)) || String(err);
  setStatus(`${what} error: ${msg}`, 'error');
}
