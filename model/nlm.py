#!/usr/bin/python
import numpy as np
from scipy.ndimage import uniform_filter


class NLM:
    """Non-Local Means Denoising

    For each pixel, searches within a (2*Ds+1)×(2*Ds+1) window for similar
    patches of size (2*ds+1)×(2*ds+1).  Similarity is measured by weighted
    Euclidean distance, and pixels are averaged with Gaussian-like weights.
    The center pixel receives the maximum weight found among all neighbors.
    """

    def __init__(self, img, ds, Ds, h, clip):
        self.img = img
        self.ds = ds    # neighbor patch half-size
        self.Ds = Ds    # search window half-size
        self.h = h      # filtering strength parameter
        self.clip = clip

    def padding(self):
        img_pad = np.pad(self.img, (self.Ds, self.Ds), 'reflect')
        return img_pad

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        """Vectorized NLM denoising.

        Instead of per-pixel loops over the search window, we iterate over
        search-window offsets and compute the block distances for ALL pixels
        simultaneously using uniform_filter (box filter) for the patch sum.

        The original calWeights iterates j in range(2*Ds+1 - 2*ds - 1),
        i.e., the number of candidate positions is (2*Ds - 2*ds)² with the
        center position skipped.  The center pixel gets weight = wmax.
        """
        img_pad = self.padding().astype(np.float64)
        raw_h, raw_w = self.img.shape
        Ds = self.Ds
        ds = self.ds
        h_sq = self.h ** 2
        patch_size = 2 * ds + 1

        # Accumulation arrays
        weighted_sum = np.zeros((raw_h, raw_w), dtype=np.float64)
        weight_sum = np.zeros((raw_h, raw_w), dtype=np.float64)
        wmax = np.zeros((raw_h, raw_w), dtype=np.float64)

        # Number of candidate offsets per axis (matching original loop range)
        n_offsets = 2 * Ds + 1 - 2 * ds - 1   # = 2*(Ds-ds)
        # Center offset index (to skip self-comparison)
        center_offset = Ds - ds

        for j in range(n_offsets):
            for i in range(n_offsets):
                if j == center_offset and i == center_offset:
                    continue  # skip self

                # dy, dx: offsets from center position in padded image
                dy = j - center_offset  # ranges from -(Ds-ds) to +(Ds-ds-1)
                dx = i - center_offset

                # Squared difference between the shifted image and the center
                # Both are views into img_pad of shape (raw_h + 2*ds, raw_w + 2*ds)
                # Center region: img_pad[Ds-ds : Ds-ds+raw_h+2*ds, ...]
                cy = Ds - ds
                cx = Ds - ds
                center_region = img_pad[cy:cy + raw_h + 2 * ds,
                                        cx:cx + raw_w + 2 * ds]
                shifted_region = img_pad[cy + dy:cy + dy + raw_h + 2 * ds,
                                         cx + dx:cx + dx + raw_w + 2 * ds]

                sq_diff = (center_region - shifted_region) ** 2

                # Sum over the patch using uniform_filter (box filter)
                # uniform_filter computes mean, so multiply by patch_size² to get sum
                # Then divide by patch_size² for the kernel normalization (original uses
                # kernel = 1/patch_size²), so the two cancel and we just need the mean
                patch_dist = uniform_filter(sq_diff, size=patch_size, mode='reflect')

                # Extract the valid region (center of the filtered result)
                dist = patch_dist[ds:ds + raw_h, ds:ds + raw_w]

                # Compute weights: w = exp(-dist / h²)
                w = np.exp(-dist / h_sq)

                # Track maximum weight
                wmax = np.maximum(wmax, w)

                # Neighbor pixel value (the center of the candidate patch)
                neighbor_val = img_pad[Ds + dy:Ds + dy + raw_h,
                                       Ds + dx:Ds + dx + raw_w]

                weighted_sum += w * neighbor_val
                weight_sum += w

        # Add center pixel with wmax weight
        center_val = img_pad[Ds:Ds + raw_h, Ds:Ds + raw_w]
        weighted_sum += wmax * center_val
        weight_sum += wmax

        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-10)

        nlm_img = (weighted_sum / weight_sum).astype(np.uint16)
        self.img = nlm_img
        return self.clipping()
