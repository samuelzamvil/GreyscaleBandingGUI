import numpy as np
import pytest
from PIL import Image

from burnbands import core


def gradient(w=256, h=4):
    """Every luminance value 0..255 present."""
    return np.tile(np.arange(256, dtype=np.uint8), (h, w // 256))


# -- breakpoints -----------------------------------------------------------

def test_even_breakpoints():
    assert core.even_breakpoints(4) == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert core.even_breakpoints(1) == [0.0, 100.0]


def test_even_breakpoints_rejects_zero():
    with pytest.raises(core.BandingError):
        core.even_breakpoints(0)


def test_validate_maps_to_luminance():
    assert core.validate_breakpoints([0, 50, 100]) == [0, 128, 255]
    assert core.validate_breakpoints([0, 100]) == [0, 255]


@pytest.mark.parametrize(
    "pcts",
    [
        [0, 100, 50],       # not increasing
        [5, 50, 100],       # doesn't start at 0
        [0, 50, 99],        # doesn't end at 100
        [0, 50, 50, 100],   # duplicate
        [0],                # too short
        [0, 0.1, 100],      # collapses to same luminance int
    ],
)
def test_validate_rejects(pcts):
    with pytest.raises(core.BandingError):
        core.validate_breakpoints(pcts)


# -- masks -----------------------------------------------------------------

def test_masks_partition_image():
    """Every pixel in exactly one band: no gaps, no overlap."""
    gray = gradient()
    thresholds = core.validate_breakpoints(core.even_breakpoints(5))
    masks = core.band_masks(gray, thresholds)
    stacked = np.sum([m.astype(int) for m in masks], axis=0)
    assert (stacked == 1).all()


def test_mask_boundary_rule():
    """Inclusive lower, exclusive upper; last band includes 255."""
    gray = np.array([[0, 127, 128, 254, 255]], dtype=np.uint8)
    masks = core.band_masks(gray, [0, 128, 255])
    assert masks[0].tolist() == [[True, True, False, False, False]]
    assert masks[1].tolist() == [[False, False, True, True, True]]


def test_coverage_sums_to_100():
    gray = gradient()
    masks = core.band_masks(gray, core.validate_breakpoints([0, 10, 60, 100]))
    cov = core.coverage(masks)
    assert sum(cov) == pytest.approx(100.0)


# -- load ------------------------------------------------------------------

def test_load_color_and_invert(tmp_path):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    path = tmp_path / "c.png"
    Image.fromarray(rgb).save(path)

    gray, valid = core.load_grayscale(path)
    assert gray.shape == (2, 2)
    assert gray[0, 0] == 76  # ITU-R 601 red weight
    assert valid.all()  # no alpha channel -> every pixel real

    inv, _ = core.load_grayscale(path, invert=True)
    assert inv[0, 0] == 255 - 76
    assert inv[1, 1] == 255


def test_load_transparency_marked_invalid(tmp_path):
    """alpha < 128 -> excluded; alpha >= 128 -> flattened onto white."""
    rgba = np.zeros((1, 3, 4), dtype=np.uint8)
    rgba[0, 0] = [0, 0, 0, 0]      # fully transparent
    rgba[0, 1] = [0, 0, 0, 127]    # mostly transparent -> excluded
    rgba[0, 2] = [0, 0, 0, 128]    # mostly opaque black -> kept, near-black
    path = tmp_path / "a.png"
    Image.fromarray(rgba, "RGBA").save(path)

    gray, valid = core.load_grayscale(path)
    assert valid.tolist() == [[False, False, True]]
    assert gray[0, 0] == 255 and gray[0, 1] == 255  # blank, not banded
    assert gray[0, 2] < 255


def test_invert_keeps_transparent_blank(tmp_path):
    rgba = np.zeros((1, 2, 4), dtype=np.uint8)
    rgba[0, 1] = [255, 255, 255, 255]  # opaque white
    path = tmp_path / "a.png"
    Image.fromarray(rgba, "RGBA").save(path)

    inv, valid = core.load_grayscale(path, invert=True)
    assert not valid[0, 0]
    assert inv[0, 0] == 255  # transparent pixel does NOT become black
    assert inv[0, 1] == 0    # real white pixel does


def test_masks_exclude_invalid_pixels():
    gray = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    valid = np.array([[True, False, True, False]])
    masks = core.band_masks(gray, [0, 128, 255], valid)
    assert masks[0].tolist() == [[True, False, False, False]]
    assert masks[1].tolist() == [[False, False, True, False]]


# -- export ----------------------------------------------------------------

def test_export_transparent(tmp_path):
    gray = gradient()
    manifest = core.export_bands(
        gray, core.even_breakpoints(3), tmp_path, dpi=300, source_name="g.png"
    )
    assert len(manifest.bands) == 3

    opaque_total = np.zeros(gray.shape, dtype=int)
    for b in manifest.bands:
        img = Image.open(tmp_path / b.filename)
        assert img.mode == "RGBA"
        assert img.size == (gray.shape[1], gray.shape[0])
        dpi = img.info["dpi"]
        assert (round(dpi[0]), round(dpi[1])) == (300, 300)
        arr = np.asarray(img)
        assert set(np.unique(arr[:, :, 3])) <= {0, 255}  # binary alpha
        assert (arr[:, :, :3][arr[:, :, 3] == 255] == 0).all()  # in-band black
        opaque_total += (arr[:, :, 3] == 255).astype(int)
    assert (opaque_total == 1).all()  # superimpose exactly once

    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.exists()
    text = manifest_file.read_text()
    for b in manifest.bands:
        assert b.filename in text


def test_export_white_bg(tmp_path):
    gray = gradient()
    manifest = core.export_bands(
        gray, [0, 50, 100], tmp_path, white_bg=True
    )
    img = Image.open(tmp_path / manifest.bands[0].filename)
    assert img.mode == "L"
    assert set(np.unique(np.asarray(img))) <= {0, 255}


def test_filenames_sorted_darkest_first(tmp_path):
    manifest = core.export_bands(gradient(), core.even_breakpoints(4), tmp_path)
    names = [b.filename for b in manifest.bands]
    assert names == sorted(names)
    assert names[0] == "band_00_L000-063.png"
    assert names[-1] == "band_03_L191-255.png"


def test_export_ignores_transparent_source(tmp_path):
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[0, 0] = [0, 0, 0, 255]        # opaque black -> darkest band
    rgba[1, 1] = [255, 255, 255, 255]  # opaque white -> lightest band
    src = tmp_path / "src.png"
    Image.fromarray(rgba, "RGBA").save(src)

    gray, valid = core.load_grayscale(src)
    out = tmp_path / "out"
    manifest = core.export_bands(gray, [0, 50, 100], out, valid=valid)
    assert manifest.transparent_pct == pytest.approx(50.0)

    opaque_total = np.zeros((2, 2), dtype=int)
    for b in manifest.bands:
        alpha = np.asarray(Image.open(out / b.filename))[:, :, 3]
        opaque_total += (alpha == 255).astype(int)
    # transparent-source pixels are in NO band; real pixels in exactly one
    assert opaque_total.tolist() == [[1, 0], [0, 1]]

    out_w = tmp_path / "out_white"
    mw = core.export_bands(gray, [0, 50, 100], out_w, valid=valid, white_bg=True)
    band0 = np.asarray(Image.open(out_w / mw.bands[0].filename))
    assert band0.tolist() == [[0, 255], [255, 255]]  # ignored pixels white


def test_overlay_shape_and_tint():
    gray = gradient()
    masks = core.band_masks(gray, [0, 128, 255])
    overlay = core.make_overlay(gray, masks, [(255, 0, 0), (0, 0, 255)])
    assert overlay.shape == (*gray.shape, 3)
    assert overlay.dtype == np.uint8
    # darkest pixel tinted toward red
    assert overlay[0, 0, 0] > overlay[0, 0, 2]
