// player.js — playback transport driven by one AudioContext and a shared beat-clock.
// Two sources share the clock: the oscillator preview synth (synth.js) and, when a render
// is fresh, the decoded rendered wav buffer. The clock is expressed in beats:
// clipBeat = passStart + (now - anchor) / secPerBeat. Loop support re-arms the active
// source at each wrap. beats->seconds via bpm (secPerBeat = 60/bpm).
//
// Source rule: three modes chosen by `sourceMode`. 'auto' uses the rendered model
// buffer when it is loaded AND fresh (else the preview synth); 'prior' uses the raw
// saw-prior buffer when loaded; 'synth' always the preview synth. Both the model and
// prior buffers start at absolute beat 0, so a pass from `fromBeat` plays them at
// offset fromBeat*secPerBeat.
import { store, bpm, notesSorted, notify } from './state.js';
import { durationBeats } from './timeline.js';
import { Synth } from './synth.js';

let ctx = null, master = null, midiGain = null, renderGain = null, priorGain = null, synth = null;
let anchor = 0, passStart = 0, secPerBeat = 0.4286;
let wrapTimer = null;

// rendered-audio sources (model mix + raw prior mix)
let renderBuffer = null;     // decoded AudioBuffer of mix.wav (null => none)
let renderSrc = null;        // live AudioBufferSourceNode for this pass
let priorBuffer = null;      // decoded AudioBuffer of prior_mix.wav (null => none)
let priorSrc = null;         // live AudioBufferSourceNode for this pass

function secPerBeatNow(){ return 60 / (bpm() || 140); }

export const player = {
  isPlaying: false,
  startBeat: 0,
  sourceMode: 'auto',        // 'auto' (model when fresh) | 'prior' | 'synth'

  init(){
    if (ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    try { ctx = new AC({ sampleRate: 44100 }); } catch { ctx = new AC(); }
    master = ctx.createGain(); master.connect(ctx.destination);
    midiGain = ctx.createGain(); midiGain.connect(master);
    renderGain = ctx.createGain(); renderGain.connect(master);
    priorGain = ctx.createGain(); priorGain.connect(master);
    synth = new Synth(ctx);
    this.setMaster(store.master);
  },
  get ctx(){ return ctx; },
  get ready(){ return !!ctx; },

  setMaster(v){ store.master = v; if (master) master.gain.value = v; },

  // ---------- rendered-audio buffer ----------
  async loadRenderWav(url){
    if (!ctx) this.init();
    // no-store: mix.wav lives at a fixed URL and changes every render — never
    // let the browser's HTTP cache answer for it.
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error('wav fetch ' + resp.status);
    const arr = await resp.arrayBuffer();
    renderBuffer = await ctx.decodeAudioData(arr);
    if (this.isPlaying && this.useRender()){ const b = this.clipBeatSafe(); this._arm(b); passStart = b; }
    notify();
    return renderBuffer;
  },
  setRenderBuffer(buf){          // used by mock/offline paths that build the buffer directly
    if (!ctx) this.init();
    renderBuffer = buf || null; notify();
  },
  clearRenderBuffer(){ renderBuffer = null; if (renderSrc){ try { renderSrc.stop(); } catch {} renderSrc = null; } notify(); },
  get hasRender(){ return !!renderBuffer; },
  get renderActive(){ return !!renderSrc; },   // rendered buffer is the live source this pass

  // ---------- prior-audio buffer (raw saw prior) ----------
  async loadPriorWav(url){
    if (!ctx) this.init();
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error('prior wav fetch ' + resp.status);
    const arr = await resp.arrayBuffer();
    priorBuffer = await ctx.decodeAudioData(arr);
    if (this.isPlaying && this.usePrior()){ const b = this.clipBeatSafe(); this._arm(b); passStart = b; }
    notify();
    return priorBuffer;
  },
  setPriorBuffer(buf){           // mock/offline paths that build the buffer directly
    if (!ctx) this.init();
    priorBuffer = buf || null; notify();
  },
  clearPriorBuffer(){ priorBuffer = null; if (priorSrc){ try { priorSrc.stop(); } catch {} priorSrc = null; } notify(); },
  hasPrior(){ return !!priorBuffer; },

  // back-compat: legacy boolean override maps onto the 'synth' / 'auto' modes.
  get forceSynth(){ return this.sourceMode === 'synth'; },

  // true when playback should use the model render rather than the preview synth
  useRender(){
    return this.sourceMode === 'auto' && !!renderBuffer &&
           !!(store.renderInfo && store.renderInfo.fresh);
  },
  // true when playback should use the raw prior buffer
  usePrior(){ return this.sourceMode === 'prior' && !!priorBuffer; },

  setSourceMode(mode){
    this.sourceMode = (mode === 'prior' || mode === 'synth') ? mode : 'auto';
    if (this.isPlaying){ const b = this.clipBeatSafe(); this._arm(b); passStart = b; }
    notify();
  },
  setForceSynth(v){ this.setSourceMode(v ? 'synth' : 'auto'); },

  previewNote(pitch){
    if (!ctx) this.init();
    if (ctx.state === 'suspended'){ ctx.resume().catch(() => {}); }
    synth.out = midiGain;
    synth.previewNote(pitch);
  },

  clipBeat(){
    if (!this.isPlaying || !ctx) return store.playhead;
    return passStart + (ctx.currentTime - anchor) / secPerBeat;
  },

  async play(fromBeat){
    if (!ctx) this.init();
    this.isPlaying = true;
    this.startBeat = fromBeat;
    store.playhead = fromBeat;
    if (ctx.state === 'suspended'){ try { await ctx.resume(); } catch {} }
    if (!this.isPlaying) return;
    this._arm(fromBeat);
    notify();
  },

  _stopSources(){
    if (synth) synth.stop();
    if (renderSrc){ try { renderSrc.onended = null; renderSrc.stop(); } catch {} renderSrc.disconnect && renderSrc.disconnect(); renderSrc = null; }
    if (priorSrc){ try { priorSrc.onended = null; priorSrc.stop(); } catch {} priorSrc.disconnect && priorSrc.disconnect(); priorSrc = null; }
  },

  _arm(fromBeat){
    this._stopSources();
    if (wrapTimer){ clearTimeout(wrapTimer); wrapTimer = null; }
    secPerBeat = secPerBeatNow();
    passStart = fromBeat;
    anchor = ctx.currentTime + 0.06;
    const loop = store.loop;
    const endBeat = loop.enabled ? loop.end_beat : durationBeats();

    if (this.useRender()){
      this._armRender(fromBeat, endBeat);
    } else if (this.usePrior()){
      this._armPrior(fromBeat, endBeat);
    } else {
      synth.out = midiGain;
      synth.schedule(anchor, fromBeat, secPerBeat, notesSorted(), endBeat, midiGain);
    }

    if (loop.enabled && loop.end_beat > loop.start_beat){
      const wallEnd = anchor + (loop.end_beat - fromBeat) * secPerBeat;
      const ms = Math.max(10, (wallEnd - ctx.currentTime) * 1000);
      wrapTimer = setTimeout(() => { if (this.isPlaying) this._arm(loop.start_beat); }, ms);
    }
  },

  // schedule an absolute-beat-0 buffer starting at absolute beat `fromBeat`
  _armBuffer(buffer, gainNode, fromBeat, endBeat){
    const offset = Math.max(0, fromBeat * secPerBeat);
    if (offset >= buffer.duration - 1e-3) return null;     // past the end: silence this pass
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(gainNode);
    const dur = Math.max(0, (endBeat - fromBeat) * secPerBeat);
    if (dur > 0) src.start(anchor, offset, dur);
    else src.start(anchor, offset);
    return src;
  },
  _armRender(fromBeat, endBeat){ renderSrc = this._armBuffer(renderBuffer, renderGain, fromBeat, endBeat); },
  _armPrior(fromBeat, endBeat){ priorSrc = this._armBuffer(priorBuffer, priorGain, fromBeat, endBeat); },

  pause(){
    const at = this.clipBeatSafe();
    this.isPlaying = false;
    if (wrapTimer){ clearTimeout(wrapTimer); wrapTimer = null; }
    store.playhead = at;
    this._stopSources();
    notify();
  },
  clipBeatSafe(){ try { return this.isPlaying ? this.clipBeat() : store.playhead; } catch { return store.playhead; } },

  stop(){
    const wasPlaying = this.isPlaying;
    this.isPlaying = false;
    if (wrapTimer){ clearTimeout(wrapTimer); wrapTimer = null; }
    this._stopSources();
    store.playhead = store.loop.enabled ? store.loop.start_beat : 0;
    notify();
    return wasPlaying;
  },

  toggle(fromBeat){ if (this.isPlaying) this.pause(); else this.play(fromBeat); },

  // re-arm mid-playback if notes/bpm/loop/source changed
  reschedule(){ if (this.isPlaying){ const b = this.clipBeatSafe(); this._arm(b); passStart = b; } },
};
