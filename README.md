# Descale

Video/Image filter to undo upscaling.

This fork only supports VapourSynth, none of the new options are mapped in AviSynth. PRs to add AviSynth support are welcome, but apart from that the AviSynth plugin can be considered frozen.

## Usage

The VapourSynth plugin itself supports every constant input format. If the format is subsampled, left-aligned chroma planes are always assumed.

```python
descale.Debilinear(clip src, int width, int height, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Debicubic(clip src, int width, int height, float b=0.0, float c=0.5, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Delanczos(clip src, int width, int height, int taps=3, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Despline16(clip src, int width, int height, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Despline36(clip src, int width, int height, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Despline64(clip src, int width, int height, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Depoint(clip src, int width, int height, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)

descale.Decustom(clip src, int width, int height, func custom_kernel, int taps=3, float blur=1.0, float[] post_conv=[], float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=0, clip ignore_mask=None, bool force=false, bool force_h=false, bool force_v=false, int opt=0)
```

The `border_handling` argument can take the following values:
- 0: Assume the image was resized with mirror padding
- 1: Assume the image was resized with zero padding
- 2: Assume the image was resized with extend padding, where the outermost row was extended infinitely far

The `opt` argument can take the following values:
- 0: Automatically decide based on CPU capabilities
- 1: No SIMD instructions
- 2: Use AVX2

The AviSynth+ plugin is used similarly, but without the `descale` namespace.
Custom kernels and ignore masks are only supported in the VapourSynth plugin.

### Custom kernels

```python
# Debilinear
core.descale.Decustom(src, w, h, custom_kernel=lambda x: 1.0 - x, taps=1)

# Delanczos
import math
def sinc(x):
    return 1.0 if x == 0 else math.sin(x * math.pi) / (x * math.pi)
taps = 3
core.descale.Decustom(src, w, h, custom_kernel=lambda x: sinc(x) * sinc(x / taps), taps=taps)
```

## How does this work?

Resampling can be described as `A x = b`.

A is an n x m matrix with `m` being the input dimension and `n` the output dimension. `x` is the original vector with `m` elements, `b` is the vector after resampling with `n` elements. We want to solve this equation for `x`.

To do this, we extend the equation with the transpose of A: `A' A x = A' b`.

`A' A` is now a banded symmetrical m x m matrix and `A' b` is a vector with `m` elements.

This enables us to use LDLT decomposition on `A' A` to get `LD L' = A' A`. `LD` and `L` are both triangular matrices.

Then we solve `LD y = A' b` with forward substitution, and finally `L' x = y` with back substitution.

We now have the original vector `x`.

## Installation

Pre-compiled wheels are available on PyPI for:

- **Windows**: x86_64
- **Linux**: x86_64 and aarch64 (glibc & musl)
- **macOS**: x86_64 (10.13+) and arm64 (11.0+)

```bash
pip install vapoursynth-descale
```

## Compilation

Building the VapourSynth plugin:

```bash
uv build --wheel
```

For the AviSynth+ plugin:

```bash
meson setup build -Dlibtype=avisynth && meson compile -C build
```
