# BurnBands

Split a grayscale image into tonal bands for laser engraving. Each band
becomes its own PNG — black pixels on a transparent background — so you can
engrave dark tones and light tones as separate passes with different power
and speed settings, then stack them back up pixel-perfect in LightBurn.

Comes as a CLI (`burnbands`) and a PySide6 GUI (`burnbands-gui`). Both run
the exact same core code, so exports are byte-identical either way.

## Install

```sh
pip install .
# or for development:
pip install -e .[dev]
```

Requires Python 3.10+. Dependencies: numpy, Pillow, PySide6.

## CLI

```sh
# 4 even bands, transparent PNGs at 300 DPI
burnbands photo.jpg -o bands/

# custom percentage breakpoints (0 and 100 required, strictly increasing)
burnbands photo.jpg -o bands/ -p 0,20,45,70,100

# invert first, white background instead of transparency, custom DPI
burnbands photo.jpg -o bands/ -b 6 --invert --white-bg --dpi 254
```

Also runnable as `python -m burnbands.cli` / `python -m burnbands.gui`.

## How banding works

1. Input (color or grayscale) is converted to grayscale using luminance
   weights (ITU-R 601). Images with transparency are flattened onto white.
2. Percentage breakpoints map to 0–255 luminance thresholds. Band *i*
   covers `lower <= value < upper`; the last band includes 255. Every pixel
   lands in exactly one band — no gaps, no overlap, no antialiasing.
3. Band 0 is the darkest. Filenames sort darkest to lightest and carry the
   luminance range: `band_00_L000-063.png`, `band_01_L064-127.png`, …
4. `manifest.json` in the output directory maps each band's index,
   percentage range, luminance range, filename, and pixel coverage.

## GUI

- **Open** a color or grayscale image; conversion happens once on load.
- Choose a **band count** and either **Even split** or **Custom
  boundaries**. Each band's lower bound is editable in custom mode, and
  spinboxes are clamped between their neighbors so boundaries always stay
  in order.
- The preview tints each band with a distinct color over the source image;
  click a band's **color swatch** to change it. Updates are live and
  debounced, computed on a downscaled copy for responsiveness.
- The **Coverage** column shows the percent of the image in each band.
- **Export** runs at full resolution and produces identical output to the
  CLI.

## LightBurn workflow

1. Export bands (default: transparent PNGs, DPI embedded — default 300).
2. In LightBurn, import each band PNG and place it at the same origin
   (they all share identical pixel dimensions, so aligned origins mean
   perfect registration).
3. Set each image's mode to **Pass-Through** so LightBurn uses the
   embedded DPI and doesn't re-dither.
4. Assign each band its own layer with dedicated power/speed — deeper burn
   for dark bands, lighter for light bands. Check `manifest.json` for which
   file covers which tonal range.

## Preview tuning

Two constants at the top of `burnbands/gui.py` control preview
responsiveness and should be tuned together: `PREVIEW_MAX_DIM` (downscale
cap, default 1024) and `DEBOUNCE_MS` (update delay, default 100).
