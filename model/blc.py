#!/usr/bin/python
import numpy as np


# Bayer pattern layout: maps pattern name to (row, col) sub-grid slices
# for each color channel [R, Gr, Gb, B]
BAYER_SLICES = {
    'rggb': {
        'r':  (slice(None, None, 2), slice(None, None, 2)),    # even row, even col
        'gr': (slice(None, None, 2), slice(1, None, 2)),       # even row, odd col
        'gb': (slice(1, None, 2),    slice(None, None, 2)),    # odd row, even col
        'b':  (slice(1, None, 2),    slice(1, None, 2)),       # odd row, odd col
    },
    'bggr': {
        'b':  (slice(None, None, 2), slice(None, None, 2)),
        'gb': (slice(None, None, 2), slice(1, None, 2)),
        'gr': (slice(1, None, 2),    slice(None, None, 2)),
        'r':  (slice(1, None, 2),    slice(1, None, 2)),
    },
    'gbrg': {
        'gb': (slice(None, None, 2), slice(None, None, 2)),
        'b':  (slice(None, None, 2), slice(1, None, 2)),
        'r':  (slice(1, None, 2),    slice(None, None, 2)),
        'gr': (slice(1, None, 2),    slice(1, None, 2)),
    },
    'grbg': {
        'gr': (slice(None, None, 2), slice(None, None, 2)),
        'r':  (slice(None, None, 2), slice(1, None, 2)),
        'b':  (slice(1, None, 2),    slice(None, None, 2)),
        'gb': (slice(1, None, 2),    slice(1, None, 2)),
    },
}


def get_bayer_slices(bayer_pattern):
    """Return a dict mapping channel name -> (row_slice, col_slice).

    Raises ValueError if pattern is unknown.
    """
    if bayer_pattern not in BAYER_SLICES:
        raise ValueError(
            f"Unknown Bayer pattern '{bayer_pattern}'. "
            f"Valid patterns: {list(BAYER_SLICES.keys())}"
        )
    return BAYER_SLICES[bayer_pattern]


class BLC:
    """Black Level Compensation

    Subtracts black level offsets per-channel and applies cross-talk
    compensation (alpha for Gr←R, beta for Gb←B).
    """

    def __init__(self, img, parameter, bayer_pattern, clip):
        self.img = img
        self.parameter = parameter      # [bl_r, bl_gr, bl_gb, bl_b, alpha, beta]
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        bl_r, bl_gr, bl_gb, bl_b, alpha, beta = self.parameter
        slices = get_bayer_slices(self.bayer_pattern)

        raw_h, raw_w = self.img.shape
        blc_img = np.empty((raw_h, raw_w), np.int16)

        # Apply black level offset to R and B first (needed for cross-talk)
        r = self.img[slices['r']] + bl_r
        b = self.img[slices['b']] + bl_b

        # Gr and Gb with cross-talk compensation
        gr = self.img[slices['gr']] + bl_gr + alpha * r / 256
        gb = self.img[slices['gb']] + bl_gb + beta * b / 256

        # Write back to output
        blc_img[slices['r']] = r
        blc_img[slices['gr']] = gr
        blc_img[slices['gb']] = gb
        blc_img[slices['b']] = b

        self.img = blc_img
        return self.clipping()
