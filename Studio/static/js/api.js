// api.js — thin fetch layer for the Studio backend. With ?mock=1 (or file://) all calls
// route to js/mock.js (an in-browser fake backed by localStorage) so the frontend runs
// with no server. Endpoint contract matches the plan's Studio/api.py table.
const params = new URLSearchParams(location.search);
export const MOCK = params.get('mock') === '1' || location.protocol === 'file:';

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

// ---------- config / models ----------
export const getConfig = ()  => MOCK ? mock.getConfig()  : j('GET', '/api/config');
export const getModels = ()  => MOCK ? mock.getModels()  : j('GET', '/api/models');

// ---------- projects ----------
export const listProjects  = ()          => MOCK ? mock.listProjects()      : j('GET', '/api/projects');
export const createProject = (body = {}) => MOCK ? mock.createProject(body)  : j('POST', '/api/projects', body);
export const getProject    = (id)        => MOCK ? mock.getProject(id)       : j('GET', `/api/projects/${id}`);
export const putProject    = (id, doc)   => MOCK ? mock.putProject(id, doc)  : j('PUT', `/api/projects/${id}`, doc);
export const renameProject    = (id, name)     => MOCK ? mock.renameProject(id, name)    : j('POST', `/api/projects/${id}/rename`, { name });
export const duplicateProject = (id, body = {}) => MOCK ? mock.duplicateProject(id, body) : j('POST', `/api/projects/${id}/duplicate`, body);
export const deleteProject    = (id)           => MOCK ? mock.deleteProject(id)           : j('DELETE', `/api/projects/${id}`);

// ---------- render (Phase 3/4; returns not_implemented in mock/Phase 1) ----------
export const startRender     = (id, opts = {}) => MOCK ? mock.startRender(id, opts)   : j('POST', `/api/projects/${id}/render`, opts);
export const getRenderStatus = (id)            => MOCK ? mock.getRenderStatus(id)     : j('GET', `/api/projects/${id}/render/status`);
export const getRenderMeta   = (id)            => MOCK ? mock.getRenderMeta(id)       : j('GET', `/api/projects/${id}/render/meta`);

// Torch-free, synchronous whole-document saw-prior render (the PRIOR playback source).
export const renderPrior = (id, opts = {}) => {
  if (MOCK) return mock.renderPrior(id, opts);
  const qs = opts.prior_mode ? `?prior_mode=${encodeURIComponent(opts.prior_mode)}` : '';
  return j('POST', `/api/projects/${id}/render-prior${qs}`);
};

// ---------- import / export ----------
export const exportProject = (id, opts = {}) => MOCK ? mock.exportProject(id, opts) : j('POST', `/api/projects/${id}/export`, opts);

// Import: the .mid is uploaded as the RAW request body (application/octet-stream) —
// python-multipart is not installed server-side, so there is no multipart form. Options
// (mode replace|merge, snap id, adopt_bpm) ride as query params. `file` is a File/Blob.
export async function importMidi(id, file, opts = {}){
  if (MOCK) return mock.importMidi(id, file, opts);
  const qs = new URLSearchParams();
  if (opts.mode) qs.set('mode', opts.mode);
  if (opts.snap) qs.set('snap', opts.snap);
  if (opts.adopt_bpm) qs.set('adopt_bpm', 'true');
  const url = `/api/projects/${id}/import-midi` + (qs.toString() ? `?${qs}` : '');
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError(r.status, data && data.error, data && data.detail);
  return data;
}

// sendBeacon alias for beforeunload flushes
export function beaconProject(id, doc){
  try {
    if (MOCK){ mock.putProject(id, doc); return true; }
    const blob = new Blob([JSON.stringify(doc)], { type: 'application/json' });
    return navigator.sendBeacon(`/api/projects/${id}`, blob);
  } catch { return false; }
}
