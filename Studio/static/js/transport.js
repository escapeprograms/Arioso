// transport.js — transport bar: play / pause / stop, loop toggle + loop-region drag on
// the ruler, BPM LCD (drag-vertically + double-click type-in), time-signature editing,
// bar:beat position readout, and master volume. BPM/time-sig live in the project doc.
import { store, bpm, timeSig, clamp, markDirty, barBeatTick, notify } from './state.js';
import { xToBeat, clampView } from './timeline.js';
import { snapBeat } from './snap.js';
import { player } from './player.js';

const $ = (id) => document.getElementById(id);
let requestStatic = () => {};

export function init(deps = {}){
  requestStatic = deps.requestStatic || (() => {});

  $('btn-play').onclick = () => playPause();
  $('btn-stop').onclick = () => { player.stop(); requestStatic(); };
  $('btn-loop').onclick = () => toggleLoop();

  $('master-vol').oninput = (e) => player.setMaster(parseFloat(e.target.value));

  wireBpm();
  wireTimeSig();
  wireRuler();
  sync();
}

export function playPause(){
  if (player.isPlaying) player.pause();
  else player.play(store.playhead);
  requestStatic();
}
export function toggleLoop(){
  store.loop.enabled = !store.loop.enabled;
  if (player.isPlaying) player.reschedule();
  sync(); requestStatic();
}
export function toggleFollow(){ store.follow = !store.follow; sync(); }

// ---------- BPM LCD ----------
function setBpm(v){
  const nv = clamp(Math.round(v * 100) / 100, 20, 300);
  if (store.doc){ store.doc.bpm = nv; markDirty(); }
  if (player.isPlaying) player.reschedule();
  sync();
}
function wireBpm(){
  const lcd = $('bpm-lcd');
  let dragging = false, sy = 0, sVal = 140;
  lcd.addEventListener('pointerdown', (e) => {
    dragging = true; sy = e.clientY; sVal = bpm(); lcd.setPointerCapture(e.pointerId); e.preventDefault();
  });
  lcd.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const dy = sy - e.clientY;               // drag up => faster
    setBpm(sVal + dy * (e.shiftKey ? 0.05 : 0.25));
  });
  lcd.addEventListener('pointerup', (e) => { dragging = false; try { lcd.releasePointerCapture(e.pointerId); } catch {} });
  lcd.addEventListener('dblclick', () => {
    const s = prompt('Tempo (BPM):', String(bpm()));
    if (s != null){ const v = parseFloat(s); if (isFinite(v)) setBpm(v); }
  });
  lcd.addEventListener('wheel', (e) => { e.preventDefault(); setBpm(bpm() + (e.deltaY < 0 ? 1 : -1)); }, { passive: false });
}

// ---------- time signature ----------
const DENS = [2, 4, 8, 16];
function setTimeSig(num, den){
  if (!store.doc) return;
  store.doc.time_sig = { num: clamp(num, 1, 32), den };
  markDirty(); sync(); requestStatic();
}
function wireTimeSig(){
  const numEl = $('ts-num'), denEl = $('ts-den');
  numEl.addEventListener('wheel', (e) => { e.preventDefault(); setTimeSig(timeSig().num + (e.deltaY < 0 ? 1 : -1), timeSig().den); }, { passive: false });
  numEl.onclick = () => setTimeSig(timeSig().num + 1, timeSig().den);
  numEl.oncontextmenu = (e) => { e.preventDefault(); setTimeSig(timeSig().num - 1, timeSig().den); };
  denEl.onclick = () => { const i = DENS.indexOf(timeSig().den); setTimeSig(timeSig().num, DENS[(i + 1) % DENS.length]); };
  denEl.oncontextmenu = (e) => { e.preventDefault(); const i = DENS.indexOf(timeSig().den); setTimeSig(timeSig().num, DENS[(i + DENS.length - 1) % DENS.length]); };
}

// ---------- ruler: loop-region paint (top half) + playhead scrub (bottom half) ----------
let rulerDrag = null;
function wireRuler(){
  const ruler = $('ruler');
  ruler.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    ruler.setPointerCapture(e.pointerId);
    const r = ruler.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const beat = Math.max(0, snapBeat(xToBeat(x)));
    if (y < r.height * 0.5){
      rulerDrag = { mode: 'loop', anchor: beat };
      store.loop.start_beat = beat; store.loop.end_beat = beat; store.loop.enabled = true;
    } else {
      rulerDrag = { mode: 'scrub' };
      store.playhead = beat; notify();
    }
    requestStatic();
  });
  ruler.addEventListener('pointermove', (e) => {
    if (!rulerDrag) return;
    const r = ruler.getBoundingClientRect();
    const beat = Math.max(0, snapBeat(xToBeat(e.clientX - r.left)));
    if (rulerDrag.mode === 'loop'){
      store.loop.start_beat = Math.min(rulerDrag.anchor, beat);
      store.loop.end_beat = Math.max(rulerDrag.anchor, beat);
    } else { store.playhead = beat; notify(); }
    requestStatic();
  });
  ruler.addEventListener('pointerup', (e) => {
    if (rulerDrag && rulerDrag.mode === 'loop' && store.loop.end_beat <= store.loop.start_beat){
      store.loop.enabled = false;    // a click (no drag) clears the loop
    }
    rulerDrag = null;
    try { ruler.releasePointerCapture(e.pointerId); } catch {}
    if (player.isPlaying) player.reschedule();
    sync(); requestStatic();
  });
}

// ---------- display sync ----------
export function sync(){
  const b = bpm();
  const intPart = Math.floor(b);
  const frac = Math.round((b - intPart) * 100);
  $('bpm-int').textContent = String(intPart);
  $('bpm-frac').textContent = '.' + String(frac).padStart(2, '0');
  $('ts-num').textContent = String(timeSig().num);
  $('ts-den').textContent = String(timeSig().den);
  $('pos-readout').textContent = barBeatTick(store.playhead);
  $('btn-loop').classList.toggle('on', store.loop.enabled);
  const pb = $('btn-play');
  pb.classList.toggle('playing', player.isPlaying);
  pb.querySelector('.glyph').textContent = player.isPlaying ? '❚❚' : '▶';
  const mv = $('master-vol');
  if (document.activeElement !== mv) mv.value = store.master;
}
