// api.js — thin fetch layer for the Labeler backend. With ?mock=1 all calls are
// routed to js/mock.js (an in-browser fake) so the frontend runs without a server.
const params = new URLSearchParams(location.search);
export const MOCK = params.get('mock') === '1';
export const SELFTEST = params.get('selftest') === '1';

let mock = null;
if (MOCK) mock = await import('./mock.js');   // top-level await (ES module)

export class ApiError extends Error {
  constructor(status, code, detail){ super(code || ('HTTP ' + status)); this.status = status; this.code = code; this.detail = detail; }
}

async function j(method, path, body){
  const opt = { method, headers: {} };
  if (body !== undefined){ opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('json') ? await r.json().catch(() => ({})) : await r.text();
  if (!r.ok) throw new ApiError(r.status, data && data.error, data && data.detail);
  return data;
}

// ---------- endpoints ----------
export const getConfig  = ()               => MOCK ? mock.getConfig()          : j('GET', '/api/config');
export const getClips   = ()               => MOCK ? mock.getClips()           : j('GET', '/api/clips');
export const getMeta    = (id)             => MOCK ? mock.getMeta(id)          : j('GET', `/api/clips/${id}/meta`);
export const getStatus  = (id)             => MOCK ? mock.getStatus(id)        : j('GET', `/api/clips/${id}/status`);
export const getNotes   = (id)             => MOCK ? mock.getNotes(id)         : j('GET', `/api/clips/${id}/notes`);
export const putNotes   = (id, doc)        => MOCK ? mock.putNotes(id, doc)    : j('PUT', `/api/clips/${id}/notes`, doc);
export const processClip= (id, opts = {})  => MOCK ? mock.processClip(id, opts): j('POST', `/api/clips/${id}/process`, opts);
export const setVerified= (id, verified)   => MOCK ? mock.setVerified(id, verified) : j('POST', `/api/clips/${id}/verified`, { verified });
export const startCompile= ()              => MOCK ? mock.startCompile()        : j('POST', '/api/compile', {});
export const compileStatus= ()             => MOCK ? mock.compileStatus()       : j('GET', '/api/compile/status');

// sendBeacon alias (POST /notes) for beforeunload flushes
export function beaconNotes(id, doc){
  try {
    if (MOCK){ mock.putNotes(id, doc); return true; }
    const blob = new Blob([JSON.stringify(doc)], { type: 'application/json' });
    return navigator.sendBeacon(`/api/clips/${id}/notes`, blob);
  } catch { return false; }
}

// ---------- media helpers ----------
// The backend nests media URLs under meta.media; older docs used meta.urls.
export const mediaOf = (meta) => (meta && (meta.media || meta.urls)) || {};

// mel tile url from meta (explicit list `mel_tiles`, or a pattern with {i}/%04d/{index}).
export function tileUrl(meta, index){
  const u = mediaOf(meta);
  if (Array.isArray(u.mel_tiles)) return u.mel_tiles[index];
  const pat = u.mel_tile_pattern;
  if (!pat) return null;
  const pad = String(index).padStart(4, '0');
  if (pat.includes('{i}'))        return pat.replace('{i}', pad);
  if (pat.includes('%04d'))       return pat.replace('%04d', pad);
  if (pat.includes('{index:04d}'))return pat.replace('{index:04d}', pad);
  if (pat.includes('{index}'))    return pat.replace('{index}', pad);
  return pat + pad + '.png';
}

// resolve the audio url for a (variant, speed). speed 1 => base wav; else stretch.
export function audioUrl(meta, variant, speed){
  const u = mediaOf(meta);
  if (speed === 1 || speed === '1') return u[variant];
  const key = String(speed);
  return (u.stretch && u.stretch[variant] && u.stretch[variant][key]) || null;
}

// fetch + decode an audio buffer (mock synthesizes silence of the right length).
export async function decodeAudio(meta, variant, speed, ctx){
  if (MOCK) return mock.decodeAudio(meta, variant, speed, ctx);
  const url = audioUrl(meta, variant, speed);
  if (!url) throw new Error(`no audio url for ${variant} @${speed}`);
  const buf = await (await fetch(url)).arrayBuffer();
  return await ctx.decodeAudioData(buf);
}

// fetch onset-strength envelope as Float32Array
export async function getOnsetEnv(meta){
  if (MOCK) return mock.getOnsetEnv(meta);
  const url = mediaOf(meta).onset_env;
  if (!url) return new Float32Array(0);
  const buf = await (await fetch(url)).arrayBuffer();
  return new Float32Array(buf);
}
