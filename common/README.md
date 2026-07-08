# common — memory palace

Project-wide shared code: the constants and audio I/O helpers that **every**
top-level package needs (the `DataSynthesizer` pipeline today, the `training`
code next). It is a peer of those packages, not owned by either.

## ⚠️ Reuse this — do not rewrite it

Before adding audio loading, wav writing, peak normalization, or a sample-rate
constant to **any** module, import it from here instead:

```python
from common.config import SR
from common.audio_io import load_mono, write_pcm16, normalize
```

Do **not** redefine `SR = 44100`, re-call `librosa.load(..., mono=True)`, or
re-implement PCM-16 writing / normalization in a new package. If a genuinely
shared helper is missing, **add it to `common/`** (and document it here) rather
than to a single consumer — that keeps training and data-gen reading/writing
audio identically. The sample rate especially must be one value across the
project: training has to read audio at the exact rate the dataset was written at.

Import mechanics: run packages as modules from the project root
(`python -m DataSynthesizer.build_dataset`, etc.), so `import common` resolves.

## Files

### config.py — canonical audio constants
- `SR = 44100` — the project sample rate; the single source of truth.
- `DEFAULT_PEAK = 0.95` — default target peak for `audio_io.normalize`.
- `PCM_SUBTYPE = "PCM_16"` — the wav subtype every output uses.
- Documents the canonical format: **mono, 16-bit PCM, `SR` Hz**.
- **Mel-spectrogram contract** — `HOP_SIZE = 512`, `N_FFT = 2048`,
  `WIN_SIZE = 2048`, `N_MELS = 128`, `FMIN = 0`, `FMAX = None`. These **must**
  match the BigVGAN-v2 vocoder checkpoint
  (`nvidia/bigvgan_v2_44khz_128band_512x`); mels computed with any other values
  feed the vocoder garbage, so they live here as the single source of truth for
  data prep, training, and inference. `DataSynthesizer/config.py` re-exports
  `HOP_SIZE as HOP` for its onset-alignment use, so the hop value is defined
  once. Truly pipeline-only constants (`FADE_MS`, `BOOKS`) still stay in
  `DataSynthesizer/config.py`.

### audio_io.py — shared audio I/O
- `load_mono(path, sr=SR)` — `librosa.load` to a mono array at `sr`.
- `write_pcm16(path, y, sr=SR)` — write a 16-bit PCM wav; returns `path`. Loads
  nothing; safe to point at an existing path to overwrite.
- `normalize(y, target_peak=DEFAULT_PEAK)` — rescale to a fixed peak so
  summed/loud audio doesn't clip; no-op on silence.

### vocoder.py — BigVGAN-v2 mel -> waveform vocoder
- The project's chosen neural vocoder. Thin wrapper over NVIDIA's BigVGAN-v2,
  vendored at `external/BigVGAN` (no PyPI package; weights pull from the HF Hub
  on first load). Checkpoint: `nvidia/bigvgan_v2_44khz_128band_512x`.
- `load_vocoder(device="cpu", use_cuda_kernel=False)` — load + assert the
  checkpoint's mel params equal the `config.py` contract (fails loudly on drift).
  `use_cuda_kernel=True` selects the fused anti-aliased-activation CUDA kernel
  (BigVGAN's main inference speedup). If the kernel can't be built, the loader
  warns and falls back to the PyTorch-native path — **still on the GPU**, just
  slower. See below to enable the real kernel.

#### Enabling the fused BigVGAN CUDA kernel (optional speedup)

`use_cuda_kernel=True` JIT-compiles a fused CUDA kernel at load time, which needs
a full CUDA **toolkit** (not just PyTorch's bundled runtime) plus a host C++
compiler. On a machine that lacks them, `torch.utils.cpp_extension.CUDA_HOME` is
`None` and the build fails, so the loader falls back to the native path. To get
the speedup (one-time, computer-level setup — several GB):

1. **CUDA Toolkit 12.x** matching your PyTorch build (`python -c "import torch;
   print(torch.version.cuda)"` — e.g. `12.4` for `torch 2.6.0+cu124`). Install
   from NVIDIA; default path `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`.
   Ensure `%CUDA_PATH%\bin` is on `PATH` so `cpp_extension.CUDA_HOME` resolves
   (it reads `CUDA_HOME` / `CUDA_PATH`); set `CUDA_HOME` to the same path if needed.
2. **MSVC C++ build tools** — "Visual Studio 2022 Build Tools" with the "Desktop
   development with C++" workload (provides `cl.exe`, required by
   `cpp_extension.load()` on Windows).
3. **Launch Jupyter from a shell where `cl.exe` is on `PATH`** — e.g. the
   "x64 Native Tools Command Prompt for VS 2022" (or run `vcvars64.bat` first),
   then start the kernel from there. Otherwise the build can't find the compiler.
4. First load with `use_cuda_kernel=True` compiles the kernel (verbose, ~a minute)
   into `external/BigVGAN/alias_free_activation/cuda/build/` and caches it; later
   loads are fast. Success = no fallback `RuntimeWarning`.

No BigVGAN source changes are needed: its hardcoded `sm_70`/`sm_80` arch flags are
forward-compatible with newer GPUs (e.g. sm_89 Ada).
- `mel_spectrogram(wav)` — mel via BigVGAN's own `meldataset.mel_spectrogram`,
  fed the `config.py` params, so it can never drift from the checkpoint. **Use
  this for any mel computation — do not re-implement an STFT.**
- `vocode(model, mel)` — run the model, returns a 1-D float waveform.
- Self-test: `python -m common.vocoder --selftest` round-trips a `Data/gt` clip.

## Consumers

- `DataSynthesizer/` — `download_audio`, `synthesizePrior`, `onset_align`,
  `build_dataset`, and `visualizations.ipynb` all import from here.
- `training/` (next) — should import `SR` and `load_mono` from here for its
  dataloader rather than defining its own.
