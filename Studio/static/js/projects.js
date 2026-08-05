// projects.js — the project browser modal: list / open / new / rename / duplicate / delete,
// wired to the toolbar "Projects" button. Rows show name, last-modified, and note count;
// the currently-open project is highlighted (its Open button reads "current"). Renaming the
// OPEN project edits store.doc.name + markDirty() (autosave persists it) rather than hitting
// the rename endpoint, so it never fights the in-memory document; renaming any other project
// goes straight to the api. Delete is a two-step inline confirm; deleting the open project
// switches away first (so the autosave flush can't recreate the doc we remove). While a
// render is running, Open (switching away) and Delete are disabled for safety.
import * as api from './api.js';
import { store, markDirty } from './state.js';
import * as rendermgr from './rendermgr.js';

const $ = (id) => document.getElementById(id);

let openProject = async () => false;
let flash = () => {};
let setStatus = () => {};

let lastList = [];
let renamingId = null;      // id whose name cell is an inline input
let creating = false;       // the "new project" input row is showing
let pendingDelete = null;   // id awaiting the second delete click
let confirmTimer = null;

export function init(deps = {}){
  openProject = deps.openProject || (async () => false);
  flash = deps.flashStatus || (() => {});
  setStatus = deps.setStatus || (() => {});

  const btn = $('btn-projects'); if (btn) btn.onclick = open;
  const close_ = $('projects-close'); if (close_) close_.onclick = close;
  const x = $('projects-cancel-x'); if (x) x.onclick = close;
  const scrim = $('projects-modal'); if (scrim) scrim.onclick = (e) => { if (e.target === scrim) close(); };
  const nw = $('proj-new'); if (nw) nw.onclick = () => { creating = true; renamingId = null; pendingDelete = null; render(); };

  const list = $('proj-list');
  if (list){
    list.onclick = onListClick;
    list.addEventListener('keydown', onListKey);
    list.addEventListener('focusout', onListBlur);
  }
}

// ---------- open / close ----------
async function open(){
  creating = false; renamingId = null; pendingDelete = null;
  const m = $('projects-modal'); if (m) m.hidden = false;
  await refresh();
}
function close(){
  const m = $('projects-modal'); if (m) m.hidden = true;
  creating = false; renamingId = null; pendingDelete = null; clearTimeout(confirmTimer);
}

// ---------- data ----------
async function refresh(){
  try { lastList = await api.listProjects(); } catch { lastList = []; }
  // The open project's unsaved edits (e.g. a just-applied rename) aren't on disk yet;
  // reflect the live in-memory name/note-count so the list matches what the editor shows.
  if (store.projectId && store.doc){
    const e = lastList.find(x => (x.id || x.project_id) === store.projectId);
    if (e){ e.name = store.doc.name; e.n_notes = (store.doc.notes || []).length; }
  }
  render();
}

// ---------- render ----------
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function render(){
  const list = $('proj-list'); if (!list) return;
  const running = rendermgr.isRunning();
  const cur = store.projectId;
  let html = '';
  if (creating){
    html += `<div class="proj-row"><div class="proj-main"><input class="proj-rename" data-new value="New Song" /></div>
      <div class="proj-actions"><button data-act="new-go">Create</button><button data-act="new-cancel">Cancel</button></div></div>`;
  }
  if (!lastList.length && !creating){
    html += '<div class="proj-empty">no projects yet — use New Project</div>';
  }
  for (const p of lastList){
    const id = p.id || p.project_id;
    const isCur = id === cur;
    const when = p.modified ? new Date(p.modified * 1000).toLocaleString() : '—';
    const n = (p.n_notes != null) ? p.n_notes : '?';
    const main = (renamingId === id)
      ? `<input class="proj-rename" data-rn="${esc(id)}" value="${esc(p.name || '')}" />`
      : `<div class="proj-name">${esc(p.name || id)}</div><div class="proj-meta">${esc(when)} · ${n} note${n === 1 ? '' : 's'}</div>`;
    const openBtn = isCur
      ? '<button class="proj-cur" disabled>current</button>'
      : `<button data-act="open" data-id="${esc(id)}"${running ? ' disabled' : ''}>Open</button>`;
    const delLbl = (pendingDelete === id) ? 'Really?' : 'Delete';
    html += `<div class="proj-row${isCur ? ' current' : ''}">
      <div class="proj-main">${main}</div>
      <div class="proj-actions">
        ${openBtn}
        <button data-act="rename" data-id="${esc(id)}">Rename</button>
        <button data-act="dup" data-id="${esc(id)}">Duplicate</button>
        <button class="danger" data-act="delete" data-id="${esc(id)}"${running ? ' disabled' : ''}>${delLbl}</button>
      </div></div>`;
  }
  list.innerHTML = html;
  const inp = list.querySelector('.proj-rename');
  if (inp){ inp.focus(); inp.select(); }
}

// ---------- events ----------
function onListClick(e){
  const btn = e.target.closest('button'); if (!btn) return;
  const act = btn.dataset.act; if (!act) return;
  const id = btn.dataset.id;
  if (act === 'open') doOpen(id);
  else if (act === 'rename'){ renamingId = id; creating = false; pendingDelete = null; render(); }
  else if (act === 'dup') doDuplicate(id);
  else if (act === 'delete') doDelete(id);
  else if (act === 'new-go'){ const i = $('proj-list').querySelector('.proj-rename[data-new]'); doCreate(i ? i.value : ''); }
  else if (act === 'new-cancel'){ creating = false; render(); }
}
function onListKey(e){
  const inp = e.target.closest('.proj-rename'); if (!inp) return;
  if (e.key === 'Enter'){
    e.preventDefault();
    if (inp.dataset.rn != null) commitRename(inp.dataset.rn, inp.value);
    else doCreate(inp.value);
  } else if (e.key === 'Escape'){
    e.preventDefault(); renamingId = null; creating = false; render();
  }
}
function onListBlur(e){
  const inp = e.target.closest('.proj-rename'); if (!inp) return;
  if (inp.dataset.rn != null && renamingId){ renamingId = null; render(); }   // cancel rename on blur
}

// ---------- actions ----------
async function doOpen(id){
  if (rendermgr.isRunning()){ flash('render in progress'); return; }
  if (id === store.projectId){ close(); return; }
  const ok = await openProject(id);
  if (ok) close(); else await refresh();
}

async function doCreate(value){
  const name = (value || '').trim() || 'New Song';
  creating = false;
  try {
    const p = await api.createProject({ name });
    if (await openProject(p.project_id)) close(); else await refresh();
  } catch { flash('create failed'); await refresh(); }
}

async function commitRename(id, value){
  const name = (value || '').trim();
  renamingId = null;
  const p = lastList.find(x => (x.id || x.project_id) === id);
  if (!name || (p && p.name === name)){ render(); return; }
  try {
    if (id === store.projectId){ store.doc.name = name; markDirty(); }   // autosave persists it
    else { await api.renameProject(id, name); }
    flash('renamed to ' + name);
  } catch { flash('rename failed'); }
  await refresh();
}

async function doDuplicate(id){
  try { const dup = await api.duplicateProject(id); flash('duplicated → ' + (dup.name || dup.project_id)); }
  catch { flash('duplicate failed'); }
  await refresh();
}

async function doDelete(id){
  if (rendermgr.isRunning()){ flash('render in progress'); return; }
  if (pendingDelete !== id){                     // first click — arm the confirm for ~3 s
    pendingDelete = id; render();
    clearTimeout(confirmTimer);
    confirmTimer = setTimeout(() => { pendingDelete = null; render(); }, 3000);
    return;
  }
  clearTimeout(confirmTimer); pendingDelete = null;
  if (id === store.projectId){
    // Switch away BEFORE deleting so the autosave flush can't recreate the doc we remove.
    try { localStorage.removeItem('studio.lastProject'); } catch {}
    const others = lastList.filter(x => (x.id || x.project_id) !== id)
      .sort((a, b) => (b.modified || 0) - (a.modified || 0));
    let switched;
    if (others.length){ switched = await openProject(others[0].id || others[0].project_id); }
    else { const np = await api.createProject({ name: 'My Song' }); switched = await openProject(np.project_id); }
    if (!switched){ flash('could not switch — delete aborted'); await refresh(); return; }
  }
  try { await api.deleteProject(id); flash('deleted'); }
  catch (err){ flash(err && err.status === 409 ? 'render in progress' : 'delete failed'); }
  await refresh();
}
