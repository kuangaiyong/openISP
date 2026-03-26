#!/usr/bin/python
import numpy as np


class DPC:
    """Dead Pixel Correction

    Detects dead (stuck) pixels by comparing each pixel against its 8
    same-color neighbors (stride-2 in Bayer space).  If ALL neighbors differ
    by more than *thres*, the pixel is replaced using either the cross-neighbor
    mean or the minimum-gradient direction.
    """

    def __init__(self, img, thres, mode, clip):
        self.img = img
        self.thres = thres
        self.mode = mode
        self.clip = clip

    def padding(self):
        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        """Vectorized dead-pixel correction.

        Pixel neighborhood layout (stride-2 on the Bayer grid):

            p1  p2  p3
            p4  p0  p5
            p6  p7  p8
        """
        img_pad = self.padding().astype(np.int32)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]

        # Extract the 9 pixel views (center + 8 neighbors)
        # img_pad has shape (raw_h+4, raw_w+4); after offset each view is (raw_h, raw_w)
        p0 = img_pad[2:2 + raw_h, 2:2 + raw_w]        # center
        p1 = img_pad[0:0 + raw_h, 0:0 + raw_w]        # top-left
        p2 = img_pad[0:0 + raw_h, 2:2 + raw_w]        # top-center
        p3 = img_pad[0:0 + raw_h, 4:4 + raw_w]        # top-right
        p4 = img_pad[2:2 + raw_h, 0:0 + raw_w]        # mid-left
        p5 = img_pad[2:2 + raw_h, 4:4 + raw_w]        # mid-right
        p6 = img_pad[4:4 + raw_h, 0:0 + raw_w]        # bot-left
        p7 = img_pad[4:4 + raw_h, 2:2 + raw_w]        # bot-center
        p8 = img_pad[4:4 + raw_h, 4:4 + raw_w]        # bot-right

        # Dead pixel mask: ALL 8 neighbors differ from center by > threshold
        dead = (
            (np.abs(p1 - p0) > self.thres) &
            (np.abs(p2 - p0) > self.thres) &
            (np.abs(p3 - p0) > self.thres) &
            (np.abs(p4 - p0) > self.thres) &
            (np.abs(p5 - p0) > self.thres) &
            (np.abs(p6 - p0) > self.thres) &
            (np.abs(p7 - p0) > self.thres) &
            (np.abs(p8 - p0) > self.thres)
        )

        # Start with the original center values
        result = p0.copy()

        if self.mode == 'mean':
            # Replace dead pixels with cross-neighbor mean
            replacement = (p2 + p4 + p5 + p7) // 4
            result[dead] = replacement[dead]

        elif self.mode == 'gradient':
            # Compute directional gradients for all pixels
            dv  = np.abs(2 * p0 - p2 - p7)   # vertical
            dh  = np.abs(2 * p0 - p4 - p5)   # horizontal
            ddl = np.abs(2 * p0 - p1 - p8)   # diagonal left (top-left to bot-right)
            ddr = np.abs(2 * p0 - p3 - p6)   # diagonal right (top-right to bot-left)

            # Replacement values for each direction (with +1 rounding like original)
            rv  = (p2 + p7 + 1) // 2
            rh  = (p4 + p5 + 1) // 2
            rdl = (p1 + p8 + 1) // 2
            rdr = (p3 + p6 + 1) // 2

            # Stack gradients and replacements to find minimum gradient direction
            gradients = np.stack([dv, dh, ddl, ddr], axis=-1)       # (H, W, 4)
            replacements = np.stack([rv, rh, rdl, rdr], axis=-1)    # (H, W, 4)

            # Index of minimum gradient at each pixel
            min_idx = np.argmin(gradients, axis=-1)                 # (H, W)

            # Gather replacement values at the min-gradient direction
            # Use advanced indexing: replacements[y, x, min_idx[y,x]]
            rows, cols = np.mgrid[0:raw_h, 0:raw_w]
            grad_replacement = replacements[rows, cols, min_idx]

            result[dead] = grad_replacement[dead]

        self.img = result.astype(np.uint16)
        return self.clipping()
