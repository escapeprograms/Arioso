// synth.js — WebAudio oscillator preview synth. One voice per note (triangle + saw ->
// lowpass -> env). Notes are scheduled absolutely against the player's shared clock
// (anchor + (start_beat - passStart) * secPerBeat), so no lookahead loop is needed for
// a single pass; the player re-arms on each loop wrap.
import { midiToHz, clamp, envDb, noteEnvCoeffs } from './state.js';

const ATTACK = 0.008, RELEASE = 0.06;
const SAW_GAIN = Math.pow(10, -8 / 20);
const EPS = 1e-4;

// Per-note energy envelope -> gain. gain(u) = PEAK_REF * 10^((db(u) - DB_REF)/20), so a
// note whose c0 sits at DB_REF plays at PEAK_REF. DB_REF is the median c0 of the gt_arky
// corpus. PEAK_REF/GAIN_MAX are tuned for level parity with the OLD flat preview (not
// Labeler's 0.15/0.3): the old formula was 0.22*(vel/127)^1.5, and the median c0 (5.4 dB)
// maps to vel velFromC0(5.4)=86, where the old peak = 0.22*(86/127)^1.5 = 0.1226. So a
// median note stays ~as loud as before. GAIN_MAX = 2*PEAK_REF clamps hot notes (+16 dB).
const DB_REF = 5.4;
const PEAK_REF = 0.122;
const GAIN_MAX = 0.244;
const CURVE_STEP_S = 0.010;     // ~10 ms wall-time sampling of the gain curve
const CURVE_MAX_SAMPLES = 256;  // cap on the Float32Array length
const CURVE_MIN_S = 0.005;      // below this the curve window is skipped (two-point fallback)

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
      // wallStart = the note's TRUE u=0 wall time (UNCLAMPED — may be in the past), so a
      // mid-entry voice enters at the correct envelope phase. startAt below is clamped.
      const wallStart = anchor + (n.start_beat - passStart) * secPerBeat;
      let wStart = anchor + (startBeat - passStart) * secPerBeat;
      const wEnd = anchor + (nEnd - passStart) * secPerBeat;
      if (wEnd <= now) continue;
      if (wStart < now) wStart = now;
      this._voice(n, wStart, wEnd, mid, wallStart);
    }
  }

  previewNote(pitch, dur = 0.35){
    if (!this.out) return;
    const now = this.ctx.currentTime + 0.005;
    this._voice({ pitch, velocity: 96 }, now, now + dur, false, now);
  }

  _voice(note, startAt, endAt, mid, wallStart){
    const ctx = this.ctx;
    const f0 = midiToHz(note.pitch);

    // gain(u): per-note energy envelope in dB -> linear gain, clamped to avoid clipping.
    // u is note-relative (0..1 over the note's full span) so it is playback-speed
    // independent; the wall duration below carries the speed scaling.
    const coeffs = noteEnvCoeffs(note);
    const gainAt = (u) => clamp(PEAK_REF * Math.pow(10, (envDb(coeffs, u) - DB_REF) / 20), EPS, GAIN_MAX);

    const oscTri = ctx.createOscillator(); oscTri.type = 'triangle'; oscTri.frequency.value = f0;
    const oscSaw = ctx.createOscillator(); oscSaw.type = 'sawtooth'; oscSaw.frequency.value = f0;
    const sawGain = ctx.createGain(); sawGain.gain.value = SAW_GAIN;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = Math.min(4 * f0, 12000); lp.Q.value = 0.5;
    const g = ctx.createGain(); g.gain.value = EPS;

    oscTri.connect(lp); oscSaw.connect(sawGain).connect(lp); lp.connect(g).connect(this.out);

    // envelope: attack -> sampled gain curve -> exponential release. u is derived per
    // sample from wall time relative to the note's TRUE start (wallStart), so it tracks
    // the real note position even for mid-entry voices. Boundary events (attack end ==
    // curve start, release start == curve end) sit exactly on the curve's endpoints, which
    // WebAudio permits — only events strictly inside a setValueCurve interval throw.
    const gp = g.gain;
    gp.cancelScheduledValues(startAt);
    const wallDur = Math.max(endAt - wallStart, 1e-4);   // full note wall span (speed-scaled)
    const relStart = Math.max(endAt - RELEASE, startAt + (mid ? 0 : ATTACK) + 0.001);
    const curveStart = mid ? startAt : startAt + ATTACK;
    const curveDur = relStart - curveStart;
    const relEnd = Math.max(endAt, relStart + 0.005);

    if (curveDur > CURVE_MIN_S){
      const N = Math.max(2, Math.min(CURVE_MAX_SAMPLES, Math.round(curveDur / CURVE_STEP_S) + 1));
      const curve = new Float32Array(N);
      for (let i = 0; i < N; i++){
        const tw = curveStart + curveDur * (i / (N - 1));
        curve[i] = gainAt(clamp((tw - wallStart) / wallDur, 0, 1));
      }
      if (mid){ gp.setValueAtTime(curve[0], startAt); }
      else { gp.setValueAtTime(EPS, startAt); gp.linearRampToValueAtTime(curve[0], startAt + ATTACK); }
      gp.setValueCurveAtTime(curve, curveStart, curveDur);
      gp.exponentialRampToValueAtTime(EPS, relEnd);
    } else {
      // degenerate short note: no curve room -> two-point peak at mid-note gain
      const peak = gainAt(0.5);
      if (mid){ gp.setValueAtTime(peak, startAt); }
      else { gp.setValueAtTime(EPS, startAt); gp.linearRampToValueAtTime(peak, startAt + ATTACK); }
      gp.setValueAtTime(peak, relStart);
      gp.exponentialRampToValueAtTime(EPS, relEnd);
    }

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
