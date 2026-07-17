// synth.js — WebAudio MIDI synth. One voice per note (triangle+saw -> lowpass -> env),
// driven by a lookahead scheduler locked to the player's shared clock so it stays
// sample-aligned with the original audio at every speed.
import { midiToHz, clamp } from './state.js';

const LOOKAHEAD_MS = 25;      // scheduler tick
const WINDOW_S = 0.12;        // how far ahead we schedule
const ATTACK = 0.008;         // 8 ms
const RELEASE = 0.06;         // 60 ms exponential
const SAW_GAIN = Math.pow(10, -8 / 20);   // -8 dB
const MAX_VOICES = 8;
const EPS = 1e-4;

export class Synth {
  constructor(ctx){
    this.ctx = ctx;
    this.timer = null;
    this.voices = [];
    this.scheduled = new Set();
  }

  // anchor/t0/s define the shared clock: wall(start_s) = anchor + (start_s - t0)/s
  start(anchor, t0, s, notes, out){
    this.anchor = anchor; this.t0 = t0; this.s = s;
    this.notes = notes;          // must be sorted by start_s
    this.out = out;
    this.scheduled = new Set();
    this._tick();
    this.timer = setInterval(() => this._tick(), LOOKAHEAD_MS);
  }

  stop(){
    if (this.timer){ clearInterval(this.timer); this.timer = null; }
    const now = this.ctx.currentTime;
    for (const v of this.voices){
      try {
        v.gain.gain.cancelScheduledValues(now);
        v.gain.gain.setTargetAtTime(0, now, 0.004);
        v.oscs.forEach(o => o.stop(now + 0.05));
      } catch {}
    }
    this.voices = [];
    this.scheduled.clear();
  }

  _tick(){
    if (!this.notes) return;
    const now = this.ctx.currentTime;
    const windowEnd = now + WINDOW_S;
    for (const n of this.notes){
      if (this.scheduled.has(n.id)) continue;
      const wStart = this.anchor + (n.start_s - this.t0) / this.s;
      const wEnd   = this.anchor + (n.end_s   - this.t0) / this.s;
      if (wEnd <= now){ this.scheduled.add(n.id); continue; }   // already elapsed
      if (wStart > windowEnd) continue;                          // future tick
      // notes already sounding at t0 begin now, mid-envelope
      const mid = wStart < now;
      this._voice(n, mid ? now : wStart, wEnd, mid);
      this.scheduled.add(n.id);
    }
    // prune finished voices
    this.voices = this.voices.filter(v => v.endAt > now - 0.1);
  }

  _voice(note, startAt, endAt, mid){
    const ctx = this.ctx;
    if (this.voices.length >= MAX_VOICES){
      const old = this.voices.shift();
      try { old.gain.gain.cancelScheduledValues(startAt); old.gain.gain.setTargetAtTime(0, startAt, 0.004); old.oscs.forEach(o => o.stop(startAt + 0.05)); } catch {}
    }
    const f0 = midiToHz(note.pitch);
    const peak = 0.25 * Math.pow(clamp(note.velocity, 1, 127) / 127, 1.5);

    const oscTri = ctx.createOscillator(); oscTri.type = 'triangle'; oscTri.frequency.value = f0;
    const oscSaw = ctx.createOscillator(); oscSaw.type = 'sawtooth'; oscSaw.frequency.value = f0;
    const sawGain = ctx.createGain(); sawGain.gain.value = SAW_GAIN;

    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = Math.min(4 * f0, 12000);
    lp.Q.value = 0.5;

    const g = ctx.createGain();
    g.gain.value = EPS;

    oscTri.connect(lp);
    oscSaw.connect(sawGain).connect(lp);
    lp.connect(g).connect(this.out);

    // envelope
    const gp = g.gain;
    gp.cancelScheduledValues(startAt);
    if (mid){
      gp.setValueAtTime(peak, startAt);
    } else {
      gp.setValueAtTime(EPS, startAt);
      gp.linearRampToValueAtTime(peak, startAt + ATTACK);
    }
    const relStart = Math.max(endAt - RELEASE, startAt + (mid ? 0 : ATTACK) + 0.001);
    gp.setValueAtTime(Math.max(peak, EPS), relStart);
    gp.exponentialRampToValueAtTime(EPS, Math.max(endAt, relStart + 0.005));

    oscTri.start(startAt); oscSaw.start(startAt);
    const stopAt = endAt + 0.02;
    oscTri.stop(stopAt); oscSaw.stop(stopAt);

    this.voices.push({ gain: g, oscs: [oscTri, oscSaw], endAt });
  }
}
