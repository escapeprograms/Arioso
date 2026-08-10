// devopts.js — the collapsible "DEV" drawer (gear icon in the toolbar). Model + checkpoint
// + vocoder selectors (from GET /api/models), a bend/quantized prior-mode toggle, and
// readouts (last render time, render device). The current selection is persisted in
// localStorage and read by rendermgr when it POSTs a render. Tolerant of both the real
// /api/models object shape ({models:[{run,checkpoints,default}], vocoders:[{name,default}],
// default_model, default_checkpoint, default_vocoder}) and the mock's.
import { store } from './state.js';

const LS = 'studio.devopts';
const $ = (id) => document.getElementById(id);

let MODELS = [];
let VOCODERS = [];
const sel = { model: null, checkpoint: null, vocoder: null, prior_mode: null };

function normalize(raw){
  let list = [], vocs = [], defModel = null, defCkpt = null, defVoc = null;
  if (Array.isArray(raw)) list = raw;
  else if (raw && Array.isArray(raw.models)){
    list = raw.models; defModel = raw.default_model; defCkpt = raw.default_checkpoint;
    if (Array.isArray(raw.vocoders)) vocs = raw.vocoders;   // absent on older backends
    defVoc = raw.default_vocoder;
  }
  return {
    models: list.map(m => ({ name: m.run || m.name, checkpoints: (m.checkpoints || []).slice(), default: !!m.default })),
    vocoders: vocs.map(v => ({ name: v.name, default: !!v.default })),
    defModel, defCkpt, defVoc,
  };
}

export function init(){
  const norm = normalize(store.models);
  const cfg = store.config || {};
  MODELS = norm.models;
  VOCODERS = norm.vocoders;

  const firstDefault = MODELS.find(m => m.default) || MODELS[0] || {};
  const defModel = norm.defModel || cfg.default_model || firstDefault.name || '7-9-adr';
  const defCkpt = norm.defCkpt || cfg.default_checkpoint || 'checkpoint_final.pt';
  const firstVocDefault = VOCODERS.find(v => v.default) || VOCODERS[0] || {};
  const defVoc = norm.defVoc || cfg.default_vocoder || firstVocDefault.name || 'ft_v2';

  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(LS)) || {}; } catch {}
  sel.model = saved.model || defModel;
  sel.checkpoint = saved.checkpoint || defCkpt;
  sel.vocoder = saved.vocoder || defVoc;
  sel.prior_mode = saved.prior_mode || cfg.default_prior_mode || 'bend';

  // older backend / mock without a vocoder scan: keep the current selection selectable
  if (!VOCODERS.length) VOCODERS = [{ name: sel.vocoder, default: true }];

  const gear = $('dev-gear'), drawer = $('dev-drawer');
  if (gear && drawer){
    gear.onclick = () => { drawer.hidden = !drawer.hidden; gear.classList.toggle('on', !drawer.hidden); };
    const closeBtn = $('dev-close');
    if (closeBtn) closeBtn.onclick = () => { drawer.hidden = true; gear.classList.remove('on'); };
  }

  populateModels();
  populateVocoders();
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

// 'hf' is the stock (un-fine-tuned) HF BigVGAN baseline; everything else is a fine-tune name.
function vocLabel(name){ return name === 'hf' ? 'stock (HF)' : name; }

function populateVocoders(){
  const vsel = $('dev-voc');
  if (!vsel) return;
  vsel.innerHTML = '';
  for (const v of VOCODERS){ const o = document.createElement('option'); o.value = v.name; o.textContent = vocLabel(v.name); vsel.appendChild(o); }
  if (!VOCODERS.find(v => v.name === sel.vocoder) && VOCODERS[0]) sel.vocoder = VOCODERS[0].name;
  vsel.value = sel.vocoder;
  vsel.onchange = () => { sel.vocoder = vsel.value; persist(); };
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
export function getOpts(){ return { model: sel.model, checkpoint: sel.checkpoint, vocoder: sel.vocoder, prior_mode: sel.prior_mode }; }

// update the drawer readouts after a completed render
export function setReadout(meta){
  const t = $('dev-last'), d = $('dev-device'), md = $('dev-rendered');
  if (t) t.textContent = new Date().toLocaleTimeString();
  if (d) d.textContent = (meta && meta.device) || '—';
  if (md) md.textContent = meta ? `${meta.model}/${meta.checkpoint} · ${meta.prior_mode} · voc ${meta.vocoder || '—'}` : '—';
}
