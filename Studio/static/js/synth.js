// synth.js — WebAudio oscillator preview synth. One voice per note (triangle + saw ->
// lowpass -> env). Notes are scheduled absolutely against the player's shared clock
// (anchor + (start_beat - passStart) * secPerBeat), so no lookahead loop is needed for
// a single pass; the player re-arms on each loop wrap.
import { midiToHz, clamp } from './state.js';

const ATTACK = 0.008, RELEASE = 0.06;
const SAW_GAIN = Math.pow(10, -8 / 20);

export class Synth {
  constructor(ctx){ this.ctx = ctx; this.voices = []; }

  // Schedule every note whose start falls in [passStart, endBeat), plus notes already
  // sounding at passStart (begun mid-envelope). out = destination GainNode.
  schedule(anchor, passStart, secPerBeat, notes, endBeat, out){
    this.out = out;
    const now = this.ctx.currentTime;
    for (const n of notes){
      const nEnd = n.start_beat + n.len_beats;
      if (nEnd <= passStart) continue;                 // fully in the past of this pass
      if (n.start_beat >= endBeat) continue;           // beyond this pass
      const mid = n.start_beat < passStart;
      const startBeat = mid ? passStart : n.start_beat;
      let wStart = anchor + (startBeat - passStart) * secPerBeat;
      const wEnd = anchor + (nEnd - passStart) * secPerBeat;
      if (wEnd <= now) continue;
      if (wStart < now) wStart = now;
      this._voice(n, wStart, wEnd, mid);
    }
  }

  previewNote(pitch, dur = 0.35){
    if (!this.out) return;
    const now = this.ctx.currentTime + 0.005;
    this._voice({ pitch, velocity: 96 }, now, now + dur, false);
  }

  _voice(note, startAt, endAt, mid){
    const ctx = this.ctx;
    const f0 = midiToHz(note.pitch);
    const peak = 0.22 * Math.pow(clamp(note.velocity || 96, 1, 127) / 127, 1.5);

    const oscTri = ctx.createOscillator(); oscTri.type = 'triangle'; oscTri.frequency.value = f0;
    const oscSaw = ctx.createOscillator(); oscSaw.type = 'sawtooth'; oscSaw.frequency.value = f0;
    const sawGain = ctx.createGain(); sawGain.gain.value = SAW_GAIN;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = Math.min(4 * f0, 12000); lp.Q.value = 0.5;
    const g = ctx.createGain(); g.gain.value = 1e-4;

    oscTri.connect(lp); oscSaw.connect(sawGain).connect(lp); lp.connect(g).connect(this.out);

    const gp = g.gain;
    gp.cancelScheduledValues(startAt);
    if (mid){ gp.setValueAtTime(peak, startAt); }
    else { gp.setValueAtTime(1e-4, startAt); gp.linearRampToValueAtTime(peak, startAt + ATTACK); }
    const relStart = Math.max(endAt - RELEASE, startAt + (mid ? 0 : ATTACK) + 0.001);
    gp.setValueAtTime(Math.max(peak, 1e-4), relStart);
    gp.exponentialRampToValueAtTime(1e-4, Math.max(endAt, relStart + 0.005));

    oscTri.start(startAt); oscSaw.start(startAt);
    const stopAt = endAt + 0.03;
    oscTri.stop(stopAt); oscSaw.stop(stopAt);
    this.voices.push({ gain: g, oscs: [oscTri, oscSaw], endAt });
    // prune finished
    const now = ctx.currentTime;
    this.voices = this.voices.filter(v => v.endAt > now - 0.2);
  }

  stop(){
    const now = this.ctx.currentTime;
    for (const v of this.voices){
      try {
        v.gain.gain.cancelScheduledValues(now);
        v.gain.gain.setTargetAtTime(0, now, 0.004);
        v.oscs.forEach(o => o.stop(now + 0.05));
      } catch {}
    }
    this.voices = [];
  }
}
