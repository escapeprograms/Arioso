// snap.js — snap engine. Grid sizes are expressed in beat units (a "beat" is one
// time-signature denominator note). A "step" is 1/4 of a beat (FL's 16-steps-per-bar
// grid in 4/4). "none" is free. Sizes respect the time signature for bar snapping.
import { store, timeSig } from './state.js';

const STEP = 0.25;   // one step = quarter of a beat

// Ordered exactly as the toolbar dropdown lists them. Keys match the backend snap ids
// (Studio/config.py SNAP_OPTIONS / timing.snap_grid) so the id round-trips through the
// project view AND is valid when sent as the import-midi `snap=` query param.
export const SNAP_OPTIONS = [
  { key: 'none',    label: '(none)'   },
  { key: 'line',    label: 'Line'     },
  { key: '1/6step', label: '1/6 step' },
  { key: '1/4step', label: '1/4 step' },
  { key: '1/3step', label: '1/3 step' },
  { key: '1/2step', label: '1/2 step' },
  { key: 'step',    label: 'Step'     },
  { key: 'beat',    label: 'Beat'     },
  { key: 'bar',     label: 'Bar'      },
];

// grid size in beats for a snap key (bar depends on the time signature)
export function gridSize(key = store.snap){
  switch (key){
    case 'none':    return 0;
    case 'line':    return STEP;           // finest visible line (== step, matches backend)
    case '1/6step': return STEP / 6;
    case '1/4step': return STEP / 4;
    case '1/3step': return STEP / 3;
    case '1/2step': return STEP / 2;
    case 'step':    return STEP;
    case 'beat':    return 1;
    case 'bar':     return timeSig().num || 4;
    default:        return STEP;
  }
}

// snap a beat value to the current grid. altFree (temporary) forces free.
export function snapBeat(beat, key = store.snap){
  if (store.altFree) return beat;
  const g = gridSize(key);
  if (!g) return beat;
  return Math.round(beat / g) * g;
}

// snap a length (never below one grid cell, or a small floor when free)
export function snapLen(lenBeats, key = store.snap){
  if (store.altFree) return Math.max(0.03125, lenBeats);
  const g = gridSize(key);
  if (!g) return Math.max(0.03125, lenBeats);
  return Math.max(g, Math.round(lenBeats / g) * g);
}

// default length for a freshly-drawn note: last-used length, snapped to the grid.
export function defaultLen(){
  const g = gridSize();
  if (!g) return store.lastLen || 1;
  return Math.max(g, Math.round((store.lastLen || 1) / g) * g);
}
