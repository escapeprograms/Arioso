// synth.js — WebAudio MIDI synth. One voice per note (triangle+saw -> lowpass -> env),
// driven by a lookahead scheduler locked to the player's shared clock so it stays
// sample-aligned with the original audio at every speed.
import { midiToHz, clamp, envDb, noteEnvCoeffs } from './state.js';

const LOOKAHEAD_MS = 25;      // scheduler tick
const WINDOW_S = 0.12;        // how far ahead we schedule
const ATTACK = 0.008;         // 8 ms
const RELEASE = 0.06;         // 60 ms exponential
const SAW_GAIN = Math.pow(10, -8 / 20);   // -8 dB
const MAX_VOICES = 8;
const EPS = 1e-4;

// Envelope -> gain mapping. gain(u) = PEAK_REF * 10^((db(u) - DB_REF)/20), so a note
// whose c0 sits at DB_REF plays at PEAK_REF. DB_REF is the median c0 of the gt_arky
// corpus; PEAK_REF makes a median note about as loud as the old flat preview at a
// typical velocity (old peak at vel 90 ~= 0.147). GAIN_MAX clamps hot notes so they
// don't clip (c0 max ~= +16 dB would otherwise map to ~0.5).
const DB_REF = 5.4;
const PEAK_REF = 0.15;
const GAIN_MAX = 0.3;
const CURVE_STEP_S = 0.010;     // ~10 ms wall-time sampling of the gain curve
const CURVE_MAX_SAMPLES = 256;  // cap on the Float32Array length
const CURVE_MIN_S = 0.005;      // below this the curve window is skipped (two-point fallback)

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
      this._voice(n, mid ? now : wStart, wEnd, mid, wStart);
      this.scheduled.add(n.id);
    }
    // prune finished voices
    this.voices = this.voices.filter(v => v.endAt > now - 0.1);
  }

  _voice(note, startAt, endAt, mid, wStart){
    const ctx = this.ctx;
    if (this.voices.length >= MAX_VOICES){
      const old = this.voices.shift();
      try { old.gain.gain.cancelScheduledValues(startAt); old.gain.gain.setTargetAtTime(0, startAt, 0.004); old.oscs.forEach(o => o.stop(startAt + 0.05)); } catch {}
    }
    const f0 = midiToHz(note.pitch);

    // gain(u): per-note energy envelope in dB -> linear gain, clamped to avoid clipping.
    // u is note-relative (0..1 over the note's full span) so it is independent of
    // playback speed; the wall duration below already carries the speed scaling.
    const coeffs = noteEnvCoeffs(note);
    const gainAt = (u) => clamp(PEAK_REF * Math.pow(10, (envDb(coeffs, u) - DB_REF) / 20), EPS, GAIN_MAX);

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

    // envelope: attack -> gain-automation curve -> exponential release. The curve is
    // sampled from gain(u) across the sustain window; u is derived per sample from wall
    // time relative to the note's wall start (wStart), so it tracks the true note
    // position even for mid-entry voices. Boundary events (attack end == curve start,
    // release start == curve end) sit exactly on the curve's endpoints, which WebAudio
    // permits — only events strictly inside a setValueCurve interval throw.
    const gp = g.gain;
    gp.cancelScheduledValues(startAt);
    const wallDur = Math.max(endAt - wStart, 1e-4);   // full note wall span (speed-scaled)
    const relStart = Math.max(endAt - RELEASE, startAt + (mid ? 0 : ATTACK) + 0.001);
    const curveStart = mid ? startAt : startAt + ATTACK;
    const curveDur = relStart - curveStart;
    const relEnd = Math.max(endAt, relStart + 0.005);

    if (curveDur > CURVE_MIN_S){
      const N = Math.max(2, Math.min(CURVE_MAX_SAMPLES, Math.round(curveDur / CURVE_STEP_S) + 1));
      const curve = new Float32Array(N);
      for (let i = 0; i < N; i++){
        const tw = curveStart + curveDur * (i / (N - 1));
        curve[i] = gainAt(clamp((tw - wStart) / wallDur, 0, 1));
      }
      if (mid){
        // enter at the current envelope position (also equals curve[0])
        gp.setValueAtTime(curve[0], startAt);
      } else {
        gp.setValueAtTime(EPS, startAt);
        gp.linearRampToValueAtTime(curve[0], startAt + ATTACK);   // 8 ms attack up to the curve's start value
      }
      gp.setValueCurveAtTime(curve, curveStart, curveDur);
      // release runs from the curve's final value (at relStart) down to EPS
      gp.exponentialRampToValueAtTime(EPS, relEnd);
    } else {
      // degenerate short note: no room for a curve window -> two-point peak at mid-note gain
      const peak = gainAt(0.5);
      if (mid){
        gp.setValueAtTime(peak, startAt);
      } else {
        gp.setValueAtTime(EPS, startAt);
        gp.linearRampToValueAtTime(peak, startAt + ATTACK);
      }
      gp.setValueAtTime(peak, relStart);
      gp.exponentialRampToValueAtTime(EPS, relEnd);
    }

    oscTri.start(startAt); oscSaw.start(startAt);
    const stopAt = endAt + 0.02;
    oscTri.stop(stopAt); oscSaw.stop(stopAt);

    this.voices.push({ gain: g, oscs: [oscTri, oscSaw], endAt });
  }
}
