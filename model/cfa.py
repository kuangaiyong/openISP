#!/usr/bin/python
import numpy as np
from scipy.ndimage import correlate


class CFA:
    """Color Filter Array Interpolation (Malvar-He-Cutler 2004)

    Demosaics a Bayer-pattern RAW image into a full RGB image using
    high-quality linear interpolation with 5x5 convolution kernels.
    Supports all four standard Bayer patterns: rggb, bggr, gbrg, grbg.
    """

    VALID_PATTERNS = {'rggb', 'bggr', 'gbrg', 'grbg'}

    def __init__(self, img, mode, bayer_pattern, clip):
        if img.ndim != 2:
            raise ValueError(f"CFA input must be 2D, got {img.ndim}D")
        if bayer_pattern not in self.VALID_PATTERNS:
            raise ValueError(f"Invalid Bayer pattern '{bayer_pattern}'")
        self.img = img
        self.mode = mode
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        """Vectorized Malvar-He-Cutler demosaicing using 2D convolution.

        The original per-pixel malvar() method implements 4 cases (r, gr, gb, b),
        each computing the two missing color channels from a 5x5 neighborhood.
        These are equivalent to 2D convolutions with specific kernels divided by 8.

        Kernel derivation from the original code:

        Case 'r' (at red pixel, compute G and B):
          G = (4*C - N - S - E - W + 2*(n+s+e+w)) / 8
          B = (6*C - 1.5*(N+S+E+W) + 2*(ne+nw+se+sw)) / 8

        Case 'b' (at blue pixel, compute R and G): same kernels as 'r' but R↔B

        Case 'gr' (at green-in-red-row, compute R and B):
          R = (5*C - ne-nw-se-sw - E-W + 0.5*(N+S) + 4*(e+w)) / 8
          B = (5*C - ne-nw-se-sw - N-S + 0.5*(E+W) + 4*(n+s)) / 8

        Case 'gb' (at green-in-blue-row, compute R and B): R↔B from 'gr'

        Where C=center, N/S/E/W=±2 offset, n/s/e/w=±1 offset, ne/nw/se/sw=±1 diag.
        """
        if self.mode != 'malvar':
            raise ValueError(f"Unsupported CFA mode: {self.mode}")

        img = self.img.astype(np.float64)
        raw_h, raw_w = img.shape

        # ---- Define the 5x5 Malvar convolution kernels (×8) ----
        # Kernel for G at R or B locations (green interpolation at color pixel)
        #   G = 4*C - N - S - E - W + 2*(n + s + e + w)
        kern_g_at_rb = np.array([
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  2,  0,  0],
            [-1,  2,  4,  2, -1],
            [ 0,  0,  2,  0,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=np.float64)

        # Kernel for B at R location (or R at B location)
        #   B = 6*C - 1.5*(N+S+E+W) + 2*(ne+nw+se+sw)
        kern_rb_at_br = np.array([
            [ 0,    0, -1.5,  0,    0],
            [ 0,    2,  0,    2,    0],
            [-1.5,  0,  6,    0, -1.5],
            [ 0,    2,  0,    2,    0],
            [ 0,    0, -1.5,  0,    0],
        ], dtype=np.float64)

        # Kernel for R at Gr location (or B at Gb location)
        #   R = 5*C - ne-nw-se-sw - E-W + 0.5*(N+S) + 4*(e+w)
        kern_rb_at_gr = np.array([
            [ 0,    0,  0.5,  0,    0],
            [ 0,   -1,  0,   -1,    0],
            [-1,    4,  5,    4,   -1],
            [ 0,   -1,  0,   -1,    0],
            [ 0,    0,  0.5,  0,    0],
        ], dtype=np.float64)

        # Kernel for B at Gr location (or R at Gb location)
        #   B = 5*C - ne-nw-se-sw - N-S + 0.5*(E+W) + 4*(n+s)
        kern_rb_at_gb = np.array([
            [ 0,    0, -1,    0,    0],
            [ 0,   -1,  4,   -1,    0],
            [ 0.5,  0,  5,    0,  0.5],
            [ 0,   -1,  4,   -1,    0],
            [ 0,    0, -1,    0,    0],
        ], dtype=np.float64)

        # ---- Apply all 4 convolutions in one pass ----
        g_at_rb  = correlate(img, kern_g_at_rb,  mode='reflect') / 8.0
        rb_at_br = correlate(img, kern_rb_at_br, mode='reflect') / 8.0
        rb_at_gr = correlate(img, kern_rb_at_gr, mode='reflect') / 8.0
        rb_at_gb = correlate(img, kern_rb_at_gb, mode='reflect') / 8.0

        # ---- Determine Bayer layout ----
        # Map bayer_pattern to the color at each of the 4 positions in a 2x2 block:
        # (even_row, even_col), (even_row, odd_col), (odd_row, even_col), (odd_row, odd_col)
        bayer_map = {
            'rggb': ('r',  'gr', 'gb', 'b'),
            'bggr': ('b',  'gb', 'gr', 'r'),
            'gbrg': ('gb', 'b',  'r',  'gr'),
            'grbg': ('gr', 'r',  'b',  'gb'),
        }
        colors = bayer_map[self.bayer_pattern]

        # ---- Build output RGB image ----
        R = np.empty((raw_h, raw_w), dtype=np.float64)
        G = np.empty((raw_h, raw_w), dtype=np.float64)
        B = np.empty((raw_h, raw_w), dtype=np.float64)

        # For each of the 4 sub-grid positions, assign R/G/B from original or convolved
        slices = [
            (slice(0, None, 2), slice(0, None, 2)),  # even row, even col
            (slice(0, None, 2), slice(1, None, 2)),  # even row, odd col
            (slice(1, None, 2), slice(0, None, 2)),  # odd row, even col
            (slice(1, None, 2), slice(1, None, 2)),  # odd row, odd col
        ]

        for (sy, sx), color in zip(slices, colors):
            if color == 'r':
                R[sy, sx] = img[sy, sx]             # original
                G[sy, sx] = g_at_rb[sy, sx]         # interpolated G at R
                B[sy, sx] = rb_at_br[sy, sx]        # interpolated B at R
            elif color == 'b':
                R[sy, sx] = rb_at_br[sy, sx]        # interpolated R at B
                G[sy, sx] = g_at_rb[sy, sx]         # interpolated G at B
                B[sy, sx] = img[sy, sx]             # original
            elif color == 'gr':
                R[sy, sx] = rb_at_gr[sy, sx]        # interpolated R at Gr
                G[sy, sx] = img[sy, sx]             # original
                B[sy, sx] = rb_at_gb[sy, sx]        # interpolated B at Gr
            elif color == 'gb':
                R[sy, sx] = rb_at_gb[sy, sx]        # interpolated R at Gb
                G[sy, sx] = img[sy, sx]             # original
                B[sy, sx] = rb_at_gr[sy, sx]        # interpolated B at Gb

        cfa_img = np.stack([R, G, B], axis=-1)
        self.img = cfa_img.astype(np.int16)
        return self.clipping()
