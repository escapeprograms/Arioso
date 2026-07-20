// player.js — playback transport driven by one AudioContext and a shared beat-clock.
// Two sources share the clock: the oscillator preview synth (synth.js) and, when a render
// is fresh, the decoded rendered wav buffer. The clock is expressed in beats:
// clipBeat = passStart + (now - anchor) / secPerBeat. Loop support re-arms the active
// source at each wrap. beats->seconds via bpm (secPerBeat = 60/bpm).
//
// Source rule: use the rendered buffer when it is loaded AND the render is fresh AND the
// user has not forced synth; otherwise the preview synth. The rendered buffer starts at
// absolute beat 0, so a pass from `fromBeat` plays it at offset fromBeat*secPerBeat.
import { store, bpm, notesSorted, notify } from './state.js';
import { durationBeats } from './timeline.js';
import { Synth } from './synth.js';

let ctx = null, master = null, midiGain = null, renderGain = null, synth = null;
let anchor = 0, passStart = 0, secPerBeat = 0.4286;
let wrapTimer = null;

// rendered-audio source
let renderBuffer = null;     // decoded AudioBuffer of mix.wav (null => none)
let renderSrc = null;        // live AudioBufferSourceNode for this pass

function secPerBeatNow(){ return 60 / (bpm() || 140); }

export const player = {
  isPlaying: false,
  startBeat: 0,
  forceSynth: false,         // user override: stay on the preview synth even when fresh

  init(){
    if (ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    try { ctx = new AC({ sampleRate: 44100 }); } catch { ctx = new AC(); }
    master = ctx.createGain(); master.connect(ctx.destination);
    midiGain = ctx.createGain(); midiGain.connect(master);
    renderGain = ctx.createGain(); renderGain.connect(master);
    synth = new Synth(ctx);
    this.setMaster(store.master);
  },
  get ctx(){ return ctx; },
  get ready(){ return !!ctx; },

  setMaster(v){ store.master = v; if (master) master.gain.value = v; },

  // ---------- rendered-audio buffer ----------
  async loadRenderWav(url){
    if (!ctx) this.init();
    const resp = await fetch(url);
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

  // true when playback should use the rendered buffer rather than the preview synth
  useRender(){
    return !!renderBuffer && !this.forceSynth &&
           !!(store.renderInfo && store.renderInfo.fresh);
  },
  setForceSynth(v){
    this.forceSynth = !!v;
    if (this.isPlaying){ const b = this.clipBeatSafe(); this._arm(b); passStart = b; }
    notify();
  },

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

  // schedule the rendered buffer starting at absolute beat `fromBeat`
  _armRender(fromBeat, endBeat){
    const offset = Math.max(0, fromBeat * secPerBeat);
    if (offset >= renderBuffer.duration - 1e-3) return;    // past the end: silence this pass
    const src = ctx.createBufferSource();
    src.buffer = renderBuffer;
    src.connect(renderGain);
    const dur = Math.max(0, (endBeat - fromBeat) * secPerBeat);
    if (dur > 0) src.start(anchor, offset, dur);
    else src.start(anchor, offset);
    renderSrc = src;
  },

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
