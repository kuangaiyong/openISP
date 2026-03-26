#!/usr/bin/python
import numpy as np
from scipy.ndimage import correlate


class EE:
    """Edge Enhancement

    Uses an edge detection filter (convolution) followed by a piecewise-linear
    edge-magnitude LUT to selectively sharpen edges in the Y (luma) channel.
    """

    def __init__(self, img, edge_filter, gain, thres, emclip):
        self.img = img
        self.edge_filter = edge_filter
        self.gain = gain          # [gain_low, gain_high]
        self.thres = thres        # [thres_low, thres_high]
        self.emclip = emclip      # [clip_min, clip_max]

    def clipping(self):
        np.clip(self.img, 0, 255, out=self.img)
        return self.img

    def _emlut_vectorized(self, em):
        """Vectorized piecewise-linear edge-magnitude LUT.

        Segments (matching the original intent, with boundary fixes):
          val <= -thres[1]             : gain[1] * val
          -thres[1] < val <= -thres[0] : 0  (dead zone, negative side)
          -thres[0] < val < thres[0]   : gain[0] * val  (low-gain pass-through)
          thres[0] <= val < thres[1]   : 0  (dead zone, positive side)
          val >= thres[1]              : gain[1] * val
        All results are divided by 256 and clipped to [emclip[0], emclip[1]].
        """
        t0, t1 = self.thres[0], self.thres[1]
        g0, g1 = self.gain[0], self.gain[1]
        clip_min, clip_max = self.emclip[0], self.emclip[1]

        em_float = em.astype(np.float64)
        result = np.zeros_like(em_float)

        # Large negative: val <= -t1
        mask_neg_large = em_float <= -t1
        result[mask_neg_large] = g1 * em_float[mask_neg_large]

        # Dead zone negative: -t1 < val <= -t0 → 0 (already zero)

        # Low-gain center: -t0 < val < t0
        mask_center = (em_float > -t0) & (em_float < t0)
        result[mask_center] = g0 * em_float[mask_center]

        # Dead zone positive: t0 <= val < t1 → 0 (already zero)

        # Large positive: val >= t1
        mask_pos_large = em_float >= t1
        result[mask_pos_large] = g1 * em_float[mask_pos_large]

        # Normalize and clip (matches original: lut / 256)
        result = result / 256.0
        np.clip(result, clip_min, clip_max, out=result)
        return result

    def execute(self):
        """Execute edge enhancement using vectorized convolution."""
        # Edge detection via 2D correlation with the 3x5 filter, then /8
        # correlate performs sum(filter * neighborhood) at each pixel
        # The original code divides by 8 after the convolution
        em_img = correlate(self.img.astype(np.float64),
                           self.edge_filter.astype(np.float64),
                           mode='reflect')
        em_img = (em_img / 8.0).astype(np.int16)

        # Apply piecewise-linear LUT to the edge map
        enhancement = self._emlut_vectorized(em_img)

        # Add enhancement to original image
        ee_img = self.img.astype(np.float64) + enhancement
        self.img = ee_img.astype(np.int16)

        return self.clipping(), em_img
