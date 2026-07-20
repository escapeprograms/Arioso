// technique.js — the Slur / Spiccato / Detache toolbar buttons and their keymap actions.
// Semantics (user spec): with ≥1 note selected, clicking a technique assigns it to ALL
// selected notes — unless every selected note already has it, in which case they all
// revert to "normal" (a toggle). No selection is a no-op (status flash). Every change goes
// through the undoable setTechnique command. Buttons carry the articulation color and light
// up when the whole current selection shares that technique.
import { store, apply, subscribe, selectedIds, noteById } from './state.js';
import * as edit from './editing.js';

let flash = () => {};

function articulations(){ return (store.config && store.config.articulations) || []; }
function keyMap(){ return (store.config && store.config.keys) || {}; }

export function init(deps = {}){
  flash = deps.flashStatus || (() => {});
  const arts = new Map(articulations().map(a => [a.name, a]));
  document.querySelectorAll('#toolbar .techbtn').forEach(b => {
    const name = b.dataset.tech;
    const a = arts.get(name);
    if (a){
      b.disabled = false;
      b.classList.remove('disabled');
      b.style.setProperty('--tech-col', a.color);
      b.title = `${label(name)} — Shift+${a.key}`;
    }
    b.onclick = () => applyTech(name);
  });
  subscribe(refresh);
  refresh();
}

function label(name){ return name.charAt(0).toUpperCase() + name.slice(1); }

// Apply a technique to the whole selection, with the revert-to-normal toggle.
export function applyTech(name){
  const ids = selectedIds();
  if (!ids.length){ flash('select note(s) first to set a technique'); return; }
  const notes = ids.map(noteById).filter(Boolean);
  if (!notes.length) return;
  let target = name;
  if (name !== 'normal' && notes.every(n => n.technique === name)) target = 'normal';
  apply(edit.setTechnique(ids, target));
}

// Keymap entry: Shift+<digit> -> articulation whose config key is that digit.
export function applyByKey(digit){
  const name = keyMap()[digit];
  if (name) applyTech(name);
}

// Light up a button when the entire current selection carries its technique.
function refresh(){
  const ids = selectedIds();
  const notes = ids.map(noteById).filter(Boolean);
  document.querySelectorAll('#toolbar .techbtn').forEach(b => {
    const name = b.dataset.tech;
    const on = notes.length > 0 && notes.every(n => n.technique === name);
    b.classList.toggle('on', on);
  });
}
