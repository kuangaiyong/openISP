#!/usr/bin/python
import numpy as np


class BNF:
    """Bilateral Noise Filtering

    Edge-preserving denoising using a 5x5 bilateral filter.
    The weight for each neighbor is the product of a fixed distance weight (dw)
    and a range weight (rw) that depends on the intensity difference.
    Range weights are quantized into 4 levels based on 3 thresholds.
    """

    def __init__(self, img, dw, rw, rthres, clip):
        self.img = img
        self.dw = dw              # 5x5 distance weight matrix
        self.rw = rw              # [rw0, rw1, rw2, rw3] range weight values
        self.rthres = rthres      # [t0, t1, t2] thresholds (t0 >= t1 >= t2)
        self.clip = clip

    def padding(self):
        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        """Vectorized bilateral filtering using shifted array views.

        Original range-weight mapping logic:
            rdiff >= rthres[0]               -> rw[0]
            rthres[1] <= rdiff < rthres[0]   -> rw[1]
            rthres[2] <= rdiff < rthres[1]   -> rw[2]
            rdiff < rthres[2]                -> rw[3]
        """
        img_pad = self.padding().astype(np.int32)
        raw_h, raw_w = self.img.shape

        # Center pixel values: shape (raw_h, raw_w)
        center = img_pad[2:2 + raw_h, 2:2 + raw_w]

        # Flatten the 5x5 kernel positions
        dw_flat = self.dw.ravel().astype(np.float64)  # (25,)

        # Build shifted views and compute range weights in one pass
        weighted_sum = np.zeros((raw_h, raw_w), dtype=np.float64)
        weight_sum = np.zeros((raw_h, raw_w), dtype=np.float64)

        for dy in range(5):
            for dx in range(5):
                # Neighbor pixel values
                neighbor = img_pad[dy:dy + raw_h, dx:dx + raw_w]

                # Absolute intensity difference
                rdiff = np.abs(neighbor - center)

                # Quantized range weight (vectorized threshold mapping)
                # Start with rw[3] (smallest diff), override upward
                rw_map = np.full((raw_h, raw_w), self.rw[3], dtype=np.float64)
                rw_map[rdiff >= self.rthres[2]] = self.rw[2]
                rw_map[rdiff >= self.rthres[1]] = self.rw[1]
                rw_map[rdiff >= self.rthres[0]] = self.rw[0]

                # Combined weight = distance weight × range weight
                w = dw_flat[dy * 5 + dx] * rw_map

                weighted_sum += w * neighbor.astype(np.float64)
                weight_sum += w

        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-10)

        bnf_img = (weighted_sum / weight_sum).astype(np.uint16)
        self.img = bnf_img
        return self.clipping()
