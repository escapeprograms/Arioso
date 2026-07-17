// player.js — playback transport. One AudioContext; original audio via a BufferSource
// and MIDI via the Synth, both on a shared clock. Buffers are lazily decoded per
// (variant, speed): a speed-s prestretched buffer is played at offset playhead/s.
import { store, clamp, notesSorted } from './state.js';
import { Synth } from './synth.js';

let ctx = null, master = null, origGain = null, midiGain = null, synth = null;
let bufferProvider = null;          // (variant, speed) => Promise<AudioBuffer>
const buffers = new Map();          // "variant|speed" -> AudioBuffer
const pending = new Map();          // in-flight decode promises

let source = null;                  // current BufferSourceNode
let anchor = 0, t0 = 0, s = 1;      // shared clock
let started = false;                // audio actually started (buffer was ready)

export const player = {
  isPlaying: false,     // intent flag (synchronous)
  startClipTime: 0,     // clip time at which playback started (for space snap-back)

  init(provider){
    if (ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    try { ctx = new AC({ sampleRate: 44100 }); } catch { ctx = new AC(); }
    master = ctx.createGain(); master.connect(ctx.destination);
    origGain = ctx.createGain(); origGain.connect(master);
    midiGain = ctx.createGain(); midiGain.connect(master);
    synth = new Synth(ctx);
    bufferProvider = provider;
    this._applyGains();
  },

  get ctx(){ return ctx; },
  get ready(){ return !!ctx; },

  _applyGains(){
    if (!ctx) return;
    const o = store.tracks.orig, m = store.tracks.midi;
    origGain.gain.value = o.mute ? 0 : o.vol;
    midiGain.gain.value = m.mute ? 0 : m.vol;
  },
  refreshGains(){ this._applyGains(); },

  _key(){ return store.variant + '|' + store.speed; },

  bufferReady(){ return buffers.has(this._key()); },

  async ensureBuffer(variant = store.variant, speed = store.speed){
    const key = variant + '|' + speed;
    if (buffers.has(key)) return buffers.get(key);
    if (pending.has(key)) return pending.get(key);
    const p = Promise.resolve(bufferProvider(variant, speed))
      .then(buf => { buffers.set(key, buf); pending.delete(key); return buf; })
      .catch(err => { pending.delete(key); throw err; });
    pending.set(key, p);
    return p;
  },

  // Kick off background decodes in priority order (current -> other speeds -> other variant).
  prefetch(){
    const others = [];
    const spOrder = [store.speed, ...(store.config.speeds || [1, 0.5, 0.25]).filter(x => x !== store.speed)];
    const varOrder = [store.variant, store.variant === 'cleaned' ? 'source' : 'cleaned'];
    for (const v of varOrder) for (const sp of spOrder) others.push([v, sp]);
    (async () => { for (const [v, sp] of others){ try { await this.ensureBuffer(v, sp); } catch {} } })();
  },

  clipTime(){
    if (!this.isPlaying || !started || !ctx) return store.playhead;
    return clamp(t0 + (ctx.currentTime - anchor) * s, 0, (store.meta && store.meta.duration_s) || 1e9);
  },

  async play(fromS){
    if (!ctx) return;
    this.isPlaying = true;          // intent flag must be synchronous (before any await)
    this.startClipTime = fromS;
    store.playhead = fromS;
    started = false;
    if (ctx.state === 'suspended'){ try { await ctx.resume(); } catch {} }
    if (!this.isPlaying) return;     // paused during resume
    store.decoding = !this.bufferReady();
    let buf;
    try { buf = await this.ensureBuffer(); }
    catch { store.decoding = false; this.isPlaying = false; return; }
    store.decoding = false;
    if (!this.isPlaying) return;   // paused while decoding
    this._start(fromS, buf);
    this.prefetch();
  },

  _start(fromS, buf){
    s = store.speed; t0 = fromS;
    anchor = ctx.currentTime + 0.06;   // small scheduling lead
    const offset = fromS / s;
    if (offset >= buf.duration){ this.isPlaying = false; return; }
    source = ctx.createBufferSource();
    source.buffer = buf;
    source.connect(origGain);
    source.onended = () => { if (this.isPlaying && source && !source._replaced) this.pause(); };
    source.start(anchor, offset);
    synth.start(anchor, t0, s, notesSorted(), midiGain);
    started = true;
  },

  pause(){
    this.isPlaying = false;
    store.decoding = false;
    if (started){ store.playhead = this.clipTime(); }
    if (source){ source._replaced = true; try { source.stop(); } catch {} source.disconnect(); source = null; }
    if (synth) synth.stop();
    started = false;
  },

  // called on clip change: stop and drop cached buffers (they belong to the old clip)
  reset(){ this.pause(); buffers.clear(); pending.clear(); },

  toggle(fromS){ if (this.isPlaying) this.pause(); else this.play(fromS); },

  setSpeed(sp){ if (this.isPlaying) this.pause(); store.speed = sp; },       // pause-first (spec)
  setVariant(v){ if (this.isPlaying) this.pause(); store.variant = v; },     // pause-first (spec)

  setVolume(track, v){ store.tracks[track].vol = v; this._applyGains(); },
  setMute(track, on){ store.tracks[track].mute = on; this._applyGains(); },
  toggleMute(track){ store.tracks[track].mute = !store.tracks[track].mute; this._applyGains(); return store.tracks[track].mute; },
};
