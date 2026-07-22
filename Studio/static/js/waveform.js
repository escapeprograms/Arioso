// waveform.js — the #wave lane. Fetches mix.peaks (the SPK1 min/max envelope the render
// pipeline precomputes), parses it, and draws a filled FL-style envelope time-aligned to
// the beat grid. Peaks are in seconds (bins_per_sec); we convert to beats with the current
// bpm so the envelope re-aligns on zoom / scroll / bpm change for free (drawWave runs every
// drawStage). Dimmed + "stale" hint when the render no longer matches the doc; empty-lane
// hint text when there is no render at all.
import { store, bpm } from './state.js';
import { beatToX, xToBeat, visibleBeatRange } from './timeline.js';

// SPK1: magic "SPK1" + uint32 sr, bins_per_sec, n_bins (LE) + interleaved int8 min,max pairs.
const MAGIC = 0x314b5053;   // "SPK1" little-endian read as uint32
let peaks = null;           // { sr, bps, nBins, min:Int8Array, max:Int8Array }

const COL = {
  bg: '#181818',
  mid: 'rgba(255,255,255,0.06)',
  fillFresh: 'rgba(90,197,79,0.42)',
  edgeFresh: 'rgba(120,220,110,0.85)',
  fillStale: 'rgba(120,140,120,0.16)',
  edgeStale: 'rgba(150,170,150,0.30)',
  hint: '#4a4a4a',
};
const PAD = 3;

export function has(){ return !!peaks; }
export function clear(){ peaks = null; }

// Parse an ArrayBuffer of a .peaks file into typed min/max arrays. Throws on bad magic.
export function parse(buf){
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('bad peaks magic');
  const sr = dv.getUint32(4, true);
  const bps = dv.getUint32(8, true);
  const nBins = dv.getUint32(12, true);
  const bytes = new Int8Array(buf, 16, Math.min(nBins * 2, (buf.byteLength - 16)));
  const mn = new Int8Array(nBins), mx = new Int8Array(nBins);
  for (let i = 0; i < nBins; i++){ mn[i] = bytes[i * 2]; mx[i] = bytes[i * 2 + 1]; }
  return { sr, bps, nBins, min: mn, max: mx };
}

// Fetch + parse a .peaks URL; installs it as the active envelope. Returns the parsed obj.
// no-store: mix.peaks lives at a fixed URL and changes every render.
export async function load(url){
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error('peaks fetch ' + resp.status);
  peaks = parse(await resp.arrayBuffer());
  return peaks;
}
// Install an already-parsed / client-built envelope (mock path).
export function setPeaks(p){ peaks = p || null; }

// ---------- draw (called by render.drawWave) ----------
export function draw(g, w, h){
  g.clearRect(0, 0, w, h);
  g.fillStyle = COL.bg; g.fillRect(0, 0, w, h);
  const cy = Math.round(h / 2);
  g.strokeStyle = COL.mid; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, cy + 0.5); g.lineTo(w, cy + 0.5); g.stroke();

  if (!peaks || !peaks.nBins){
    g.fillStyle = COL.hint; g.font = '10px ui-sans-serif'; g.textBaseline = 'middle';
    g.fillText('waveform — press Render to preview audio', 8, cy);
    return;
  }

  const fresh = !!(store.renderInfo && store.renderInfo.fresh);
  const spb = 60 / (bpm() || 140);      // seconds per beat (== render bpm while fresh)
  const bps = peaks.bps, nBins = peaks.nBins;
  const amp = (h / 2) - PAD;

  // Per-pixel-column envelope over the visible range.
  const [vb0] = visibleBeatRange();
  const x0 = Math.max(0, Math.floor(beatToX(Math.max(0, vb0))));
  const tops = [], bots = []; let any = false;
  for (let x = x0; x <= w; x++){
    const bL = xToBeat(x), bR = xToBeat(x + 1);
    if (bR <= 0) continue;
    let i0 = Math.floor(Math.max(0, bL) * spb * bps);
    let i1 = Math.ceil(bR * spb * bps);
    if (i0 >= nBins) break;              // past the end of the audio
    if (i1 > nBins) i1 = nBins;
    if (i1 <= i0) i1 = i0 + 1;
    let mn = 127, mx = -127;
    for (let i = i0; i < i1; i++){ const a = peaks.min[i], b = peaks.max[i]; if (a < mn) mn = a; if (b > mx) mx = b; }
    if (mx < mn){ continue; }
    tops.push([x, cy - (mx / 127) * amp]);
    bots.push([x, cy - (mn / 127) * amp]);
    any = true;
  }
  if (!any){
    if (!fresh){ g.fillStyle = COL.hint; g.font = '10px ui-sans-serif'; g.textBaseline = 'middle'; g.fillText('render out of view', 8, cy); }
    return;
  }

  g.beginPath();
  g.moveTo(tops[0][0], tops[0][1]);
  for (let k = 1; k < tops.length; k++) g.lineTo(tops[k][0], tops[k][1]);
  for (let k = bots.length - 1; k >= 0; k--) g.lineTo(bots[k][0], bots[k][1]);
  g.closePath();
  g.fillStyle = fresh ? COL.fillFresh : COL.fillStale; g.fill();

  // top/bottom edge stroke for definition
  g.strokeStyle = fresh ? COL.edgeFresh : COL.edgeStale; g.lineWidth = 1;
  g.beginPath();
  g.moveTo(tops[0][0], tops[0][1]);
  for (let k = 1; k < tops.length; k++) g.lineTo(tops[k][0], tops[k][1]);
  g.stroke();
  g.beginPath();
  g.moveTo(bots[0][0], bots[0][1]);
  for (let k = 1; k < bots.length; k++) g.lineTo(bots[k][0], bots[k][1]);
  g.stroke();

  if (!fresh){
    g.fillStyle = 'rgba(224,162,79,0.9)'; g.font = '600 9px ui-sans-serif'; g.textBaseline = 'top';
    g.fillText('STALE — re-render', 8, 4);
  }
}
