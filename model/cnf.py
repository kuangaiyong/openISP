#!/usr/bin/python
import numpy as np



class CNF:
    """Chroma Noise Filtering

    Detects and corrects chroma noise on R and B pixels in the Bayer domain.
    Green pixels are passed through unchanged.

    The algorithm:
      1. CND (Chroma Noise Detection): compute local averages of G, same-color
         (C1), and cross-color (C2) channels.  A pixel is "noisy" if both the
         center and its same-color average exceed the G and cross-color averages
         by more than a threshold.
      2. CNC (Chroma Noise Correction): blend the noisy pixel toward the local
         average using a damping factor (based on AWB gain) and fade factors
         (based on signal level).
    """

    VALID_PATTERNS = {'rggb', 'bggr', 'gbrg', 'grbg'}

    def __init__(self, img, bayer_pattern, thres, gain, clip):
        if img.ndim != 2:
            raise ValueError(f"CNF input must be 2D, got {img.ndim}D")
        if bayer_pattern not in self.VALID_PATTERNS:
            raise ValueError(f"Invalid Bayer pattern '{bayer_pattern}'")
        self.img = img
        self.bayer_pattern = bayer_pattern
        self.thres = thres
        self.gain = gain          # [r_gain, gr_gain, gb_gain, b_gain]
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    @staticmethod
    def _piecewise_fade(val, thresholds, values, default):
        """Vectorized piecewise-constant function.

        Args:
            val: ndarray of values to evaluate
            thresholds: list of upper bounds [t0, t1, ...] in ascending order
            values: list of corresponding output values [v0, v1, ...]
            default: output for val > last threshold
        """
        result = np.full_like(val, default, dtype=np.float64)
        # Apply in reverse order so the smallest threshold wins
        for t, v in reversed(list(zip(thresholds, values))):
            result[val <= t] = v
        return result

    def _process_color_channel(self, img_pad, raw_h, raw_w,
                               color, cy, cx, gy1, gx1, gy2, gx2,
                               c2y, c2x):
        """Process one color channel (R or B) for noise detection and correction.

        Args:
            img_pad: padded image
            raw_h, raw_w: original size
            color: 'r' or 'b'
            cy, cx: sub-grid parity of this color
            gy1, gx1: first green sub-grid parity
            gy2, gx2: second green sub-grid parity
            c2y, c2x: cross-color sub-grid parity
        """
        pad = 4
        half_h = raw_h // 2
        half_w = raw_w // 2
        radius = 4

        # Extract center pixel values for this color channel
        center = img_pad[pad + cy:pad + cy + 2 * half_h:2,
                         pad + cx:pad + cx + 2 * half_w:2].astype(np.float64)

        # Compute avgG: sum of green pixels (two sub-grids) / 40
        avgG = np.zeros((half_h, half_w), dtype=np.float64)
        # Green sub-grid 1
        for di in range(-radius, radius):
            for dj in range(-radius, radius):
                ay = (pad + cy + di) % 2
                ax = (pad + cx + dj) % 2
                if (ay == gy1 and ax == gx1) or (ay == gy2 and ax == gx2):
                    y_start = pad + cy + di
                    x_start = pad + cx + dj
                    if y_start >= 0 and x_start >= 0:
                        neighbor = img_pad[y_start:y_start + 2 * half_h:2,
                                           x_start:x_start + 2 * half_w:2]
                        avgG += neighbor.astype(np.float64)
        avgG = avgG / 40.0

        # Compute avgC1: same color channel / 25
        avgC1 = np.zeros((half_h, half_w), dtype=np.float64)
        for di in range(-radius, radius):
            for dj in range(-radius, radius):
                ay = (pad + cy + di) % 2
                ax = (pad + cx + dj) % 2
                if ay == cy and ax == cx:
                    y_start = pad + cy + di
                    x_start = pad + cx + dj
                    neighbor = img_pad[y_start:y_start + 2 * half_h:2,
                                       x_start:x_start + 2 * half_w:2]
                    avgC1 += neighbor.astype(np.float64)
        avgC1 = avgC1 / 25.0

        # Compute avgC2: cross color channel / 16
        avgC2 = np.zeros((half_h, half_w), dtype=np.float64)
        for di in range(-radius, radius):
            for dj in range(-radius, radius):
                ay = (pad + cy + di) % 2
                ax = (pad + cx + dj) % 2
                if ay == c2y and ax == c2x:
                    y_start = pad + cy + di
                    x_start = pad + cx + dj
                    neighbor = img_pad[y_start:y_start + 2 * half_h:2,
                                       x_start:x_start + 2 * half_w:2]
                    avgC2 += neighbor.astype(np.float64)
        avgC2 = avgC2 / 16.0

        # ---- Chroma Noise Detection (CND) ----
        max_g_c2 = np.maximum(avgG, avgC2)
        is_noise = (
            (center > avgG + self.thres) &
            (center > avgC2 + self.thres) &
            (avgC1 > avgG + self.thres) &
            (avgC1 > avgC2 + self.thres)
        )

        # ---- Chroma Noise Correction (CNC) ----
        # Damping factor based on AWB gain
        if color == 'r':
            gain_val = self.gain[0]
        else:
            gain_val = self.gain[3]

        if gain_val <= 1.0:
            dampFactor = 1.0
        elif gain_val <= 1.2:
            dampFactor = 0.5
        else:
            dampFactor = 0.3

        signalGap = center - max_g_c2
        chromaCorrected = max_g_c2 + dampFactor * signalGap

        # Signal meter
        if color == 'r':
            signalMeter = 0.299 * avgC1 + 0.587 * avgG + 0.114 * avgC2
        else:
            signalMeter = 0.299 * avgC2 + 0.587 * avgG + 0.114 * avgC1

        # fade1: piecewise function of signalMeter
        fade1 = self._piecewise_fade(
            signalMeter,
            thresholds=[30, 50, 70, 100, 150, 200, 250],
            values=[1.0, 0.9, 0.8, 0.7, 0.6, 0.3, 0.1],
            default=0.0
        )

        # fade2: piecewise function of avgC1
        fade2 = self._piecewise_fade(
            avgC1,
            thresholds=[30, 50, 70, 100, 150, 200],
            values=[1.0, 0.9, 0.8, 0.6, 0.5, 0.3],
            default=0.0
        )

        fadeTot = fade1 * fade2
        corrected = (1 - fadeTot) * center + fadeTot * chromaCorrected

        # Apply correction only to noise pixels; keep original otherwise
        result = np.where(is_noise, corrected, center)
        return result

    def execute(self):
        """Vectorized chroma noise filtering."""
        img_pad = np.pad(self.img, ((4, 4), (4, 4)), 'reflect')
        img_pad = img_pad.astype(np.float64)
        raw_h, raw_w = self.img.shape
        cnf_img = self.img.astype(np.float64).copy()

        # Determine sub-grid parities based on Bayer pattern
        # pad=4 is even, so parity at padded position (4+0, 4+0) = parity at (0,0) in original
        # For rggb: (0,0)=R(even,even), (0,1)=Gr(even,odd), (1,0)=Gb(odd,even), (1,1)=B(odd,odd)
        bayer_map = {
            'rggb': {'r': (0, 0), 'gr': (0, 1), 'gb': (1, 0), 'b': (1, 1)},
            'bggr': {'b': (0, 0), 'gb': (0, 1), 'gr': (1, 0), 'r': (1, 1)},
            'gbrg': {'gb': (0, 0), 'b': (0, 1), 'r': (1, 0), 'gr': (1, 1)},
            'grbg': {'gr': (0, 0), 'r': (0, 1), 'b': (1, 0), 'gb': (1, 1)},
        }
        positions = bayer_map[self.bayer_pattern]
        ry, rx = positions['r']
        by, bx = positions['b']
        gry, grx = positions['gr']
        gby, gbx = positions['gb']

        # Process R channel
        r_result = self._process_color_channel(
            img_pad, raw_h, raw_w,
            color='r', cy=ry, cx=rx,
            gy1=gry, gx1=grx, gy2=gby, gx2=gbx,
            c2y=by, c2x=bx
        )
        cnf_img[ry::2, rx::2] = r_result

        # Process B channel
        b_result = self._process_color_channel(
            img_pad, raw_h, raw_w,
            color='b', cy=by, cx=bx,
            gy1=gry, gx1=grx, gy2=gby, gx2=gbx,
            c2y=ry, c2x=rx
        )
        cnf_img[by::2, bx::2] = b_result

        # Green pixels are unchanged (already in cnf_img from the copy)
        self.img = cnf_img.astype(np.uint16)
        return self.clipping()
