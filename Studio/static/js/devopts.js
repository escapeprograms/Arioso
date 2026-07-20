// devopts.js — the collapsible "DEV" drawer (gear icon in the toolbar). Model + checkpoint
// selectors (from GET /api/models), a bend/quantized prior-mode toggle, and readouts (last
// render time, render device). The current selection is persisted in localStorage and read
// by rendermgr when it POSTs a render. Tolerant of both the real /api/models object shape
// ({models:[{run,checkpoints,default}], default_model, default_checkpoint}) and the mock's.
import { store } from './state.js';

const LS = 'studio.devopts';
const $ = (id) => document.getElementById(id);

let MODELS = [];
const sel = { model: null, checkpoint: null, prior_mode: null };

function normalize(raw){
  let list = [], defModel = null, defCkpt = null;
  if (Array.isArray(raw)) list = raw;
  else if (raw && Array.isArray(raw.models)){ list = raw.models; defModel = raw.default_model; defCkpt = raw.default_checkpoint; }
  return {
    models: list.map(m => ({ name: m.run || m.name, checkpoints: (m.checkpoints || []).slice(), default: !!m.default })),
    defModel, defCkpt,
  };
}

export function init(){
  const norm = normalize(store.models);
  const cfg = store.config || {};
  MODELS = norm.models;

  const firstDefault = MODELS.find(m => m.default) || MODELS[0] || {};
  const defModel = norm.defModel || cfg.default_model || firstDefault.name || '7-9-adr';
  const defCkpt = norm.defCkpt || cfg.default_checkpoint || 'checkpoint_final.pt';

  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(LS)) || {}; } catch {}
  sel.model = saved.model || defModel;
  sel.checkpoint = saved.checkpoint || defCkpt;
  sel.prior_mode = saved.prior_mode || cfg.default_prior_mode || 'bend';

  const gear = $('dev-gear'), drawer = $('dev-drawer');
  if (gear && drawer){
    gear.onclick = () => { drawer.hidden = !drawer.hidden; gear.classList.toggle('on', !drawer.hidden); };
    const closeBtn = $('dev-close');
    if (closeBtn) closeBtn.onclick = () => { drawer.hidden = true; gear.classList.remove('on'); };
  }

  populateModels();
  wirePrior();
  persist();
}

function populateModels(){
  const msel = $('dev-model');
  if (!msel) return;
  msel.innerHTML = '';
  for (const m of MODELS){ const o = document.createElement('option'); o.value = m.name; o.textContent = m.name; msel.appendChild(o); }
  if (!MODELS.find(m => m.name === sel.model) && MODELS[0]) sel.model = MODELS[0].name;
  msel.value = sel.model;
  msel.onchange = () => { sel.model = msel.value; fillCkpts(); persist(); };
  fillCkpts();
}

function fillCkpts(){
  const csel = $('dev-ckpt');
  if (!csel) return;
  const m = MODELS.find(x => x.name === sel.model);
  const ckpts = (m && m.checkpoints && m.checkpoints.length) ? m.checkpoints : [sel.checkpoint].filter(Boolean);
  csel.innerHTML = '';
  for (const c of ckpts){ const o = document.createElement('option'); o.value = c; o.textContent = c; csel.appendChild(o); }
  if (!ckpts.includes(sel.checkpoint)) sel.checkpoint = ckpts[0] || sel.checkpoint;
  csel.value = sel.checkpoint;
  csel.onchange = () => { sel.checkpoint = csel.value; persist(); };
}

function wirePrior(){
  document.querySelectorAll('#dev-drawer [data-prior]').forEach(b => {
    b.onclick = () => { sel.prior_mode = b.dataset.prior; syncPrior(); persist(); };
  });
  syncPrior();
}
function syncPrior(){
  document.querySelectorAll('#dev-drawer [data-prior]').forEach(b => b.classList.toggle('on', b.dataset.prior === sel.prior_mode));
}

function persist(){ try { localStorage.setItem(LS, JSON.stringify(sel)); } catch {} }

// consumed by rendermgr for the POST body
export function getOpts(){ return { model: sel.model, checkpoint: sel.checkpoint, prior_mode: sel.prior_mode }; }

// update the drawer readouts after a completed render
export function setReadout(meta){
  const t = $('dev-last'), d = $('dev-device'), md = $('dev-rendered');
  if (t) t.textContent = new Date().toLocaleTimeString();
  if (d) d.textContent = (meta && meta.device) || '—';
  if (md) md.textContent = meta ? `${meta.model}/${meta.checkpoint} · ${meta.prior_mode}` : '—';
}
