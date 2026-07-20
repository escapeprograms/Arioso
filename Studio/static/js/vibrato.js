// vibrato.js — the selection inspector strip (above the status bar). Visible whenever ≥1
// note is selected; three mini sliders edit vibrato depth / rate / onset. Each shows the
// common value across the selection, or "mixed" when they differ. Dragging a slider live-
// updates its readout; releasing it writes one undoable setVibrato (a partial patch that
// touches only that field) to every selected note. A note with depth>0 also gets a small
// "V" glyph on its rect (drawn by render.js).
import { store, apply, subscribe, selectedIds, noteById } from './state.js';
import * as edit from './editing.js';

const FIELDS = [
  { id: 'vib-depth', key: 'depth_semitones', dec: 2, unit: ' st' },
  { id: 'vib-rate',  key: 'rate_hz',         dec: 1, unit: ' Hz' },
  { id: 'vib-onset', key: 'onset_beats',     dec: 2, unit: ' b'  },
];

let panel = null;

export function init(){
  panel = document.getElementById('inspector');
  for (const f of FIELDS){
    const slider = document.getElementById(f.id);
    const valEl = document.getElementById(f.id + '-v');
    if (!slider) continue;
    slider.addEventListener('input', () => { valEl.textContent = fmt(slider.value, f); });
    slider.addEventListener('change', () => {
      const ids = selectedIds();
      if (ids.length) apply(edit.setVibrato(ids, { [f.key]: parseFloat(slider.value) }));
    });
  }
  subscribe(refresh);
  refresh();
}

function fmt(v, f){ return Number(v).toFixed(f.dec) + f.unit; }
function mean(a){ return a.reduce((s, v) => s + v, 0) / a.length; }

function refresh(){
  if (!panel) return;
  const notes = selectedIds().map(noteById).filter(Boolean);
  panel.hidden = notes.length === 0;
  if (!notes.length) return;
  for (const f of FIELDS){
    const slider = document.getElementById(f.id);
    const valEl = document.getElementById(f.id + '-v');
    if (!slider) continue;
    if (slider === document.activeElement) continue;   // don't fight an active drag
    const vals = notes.map(n => (n.vibrato || {})[f.key] ?? 0);
    const allEq = vals.every(v => v === vals[0]);
    slider.value = allEq ? vals[0] : mean(vals);
    valEl.textContent = allEq ? fmt(vals[0], f) : 'mixed';
    valEl.classList.toggle('mixed', !allEq);
  }
}
