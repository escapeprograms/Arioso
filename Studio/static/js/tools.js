// tools.js — tool state (draw/select/delete/slice) + toolbar button and snap-dropdown
// wiring. Kept separate from interact.js so the pointer state machine only reads store.tool.
import { store, notify } from './state.js';
import { SNAP_OPTIONS } from './snap.js';

export const TOOLS = ['draw', 'select', 'delete', 'slice'];

let requestStatic = () => {};

export function setTool(t){
  if (!TOOLS.includes(t)) return;
  store.tool = t;
  syncToolButtons();
  const roll = document.getElementById('grid');
  if (roll) roll.style.cursor = t === 'draw' ? 'crosshair' : t === 'delete' ? 'not-allowed' : t === 'slice' ? 'col-resize' : 'default';
  notify();
}
export function setSnap(key){
  store.snap = key;
  const sel = document.getElementById('snap-select');
  if (sel && sel.value !== key) sel.value = key;
  requestStatic();
  notify();
}

function syncToolButtons(){
  document.querySelectorAll('#toolbar [data-tool]').forEach(b => {
    b.classList.toggle('on', b.dataset.tool === store.tool);
  });
}

export function init(deps = {}){
  requestStatic = deps.requestStatic || (() => {});
  // tool buttons
  document.querySelectorAll('#toolbar [data-tool]').forEach(b => {
    b.onclick = () => setTool(b.dataset.tool);
  });
  // snap dropdown (populate)
  const sel = document.getElementById('snap-select');
  if (sel){
    sel.innerHTML = '';
    for (const o of SNAP_OPTIONS){
      const opt = document.createElement('option');
      opt.value = o.key; opt.textContent = o.label;
      sel.appendChild(opt);
    }
    sel.value = store.snap;
    sel.onchange = () => setSnap(sel.value);
  }
  syncToolButtons();
}
