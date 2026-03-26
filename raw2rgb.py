#!/usr/bin/python
"""
raw2rgb.py - 简化的 RAW 到 RGB 图像处理入口

用法:
    python raw2rgb.py input.RAW                          # 自动推断所有参数
    python raw2rgb.py input.RAW -w 1920 -h 1080          # 指定宽高
    python raw2rgb.py input.RAW --bits 12 --bayer bggr   # 指定位深和 Bayer 模式
    python raw2rgb.py input.RAW -o output.png             # 指定输出文件
    python raw2rgb.py input.RAW --config config.csv       # 使用自定义配置文件
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt

from model.dpc import DPC
from model.blc import BLC
from model.aaf import AAF
from model.awb import WBGC
from model.cnf import CNF
from model.cfa import CFA
from model.gac import GC
from model.ccm import CCM
from model.csc import CSC
from model.bnf import BNF
from model.eeh import EE
from model.fcs import FCS
from model.bcc import BCC
from model.hsc import HSC
from model.nlm import NLM


# ---------- 常见分辨率数据库（宽 × 高） ----------
COMMON_RESOLUTIONS = [
    (160, 120),    # QQVGA
    (320, 240),    # QVGA
    (640, 480),    # VGA
    (800, 600),    # SVGA
    (1024, 768),   # XGA
    (1280, 720),   # HD 720p
    (1280, 960),   # SXGA-
    (1280, 1024),  # SXGA
    (1600, 1200),  # UXGA
    (1920, 1080),  # Full HD
    (1920, 1200),  # WUXGA
    (2048, 1536),  # QXGA
    (2560, 1440),  # QHD
    (2592, 1944),  # 5MP
    (3264, 2448),  # 8MP
    (3840, 2160),  # 4K UHD
    (4000, 3000),  # 12MP
    (4032, 3024),  # 12MP (iPhone)
    (4096, 2160),  # DCI 4K
    (5120, 2880),  # 5K
    (7680, 4320),  # 8K UHD
]


class RawFileAnalyzer:
    """从 RAW 文件中自动推断图像参数"""

    @staticmethod
    def guess_resolution(file_size_bytes):
        """根据文件大小推断分辨率和字节大小

        Args:
            file_size_bytes: RAW 文件大小（字节）

        Returns:
            list: 候选 (width, height, bytes_per_pixel) 列表
        """
        candidates = []
        for bpp in [2, 1]:  # 优先尝试 uint16(2字节), 再尝试 uint8(1字节)
            total_pixels = file_size_bytes / bpp
            if total_pixels != int(total_pixels):
                continue
            total_pixels = int(total_pixels)

            for w, h in COMMON_RESOLUTIONS:
                if w * h == total_pixels:
                    candidates.append((w, h, bpp))

            # 如果没有精确匹配，尝试宽高互换
            for w, h in COMMON_RESOLUTIONS:
                if h * w == total_pixels and (h, w, bpp) not in candidates:
                    candidates.append((h, w, bpp))

        return candidates

    @staticmethod
    def guess_bit_depth(raw_data):
        """根据数据最大值推断实际位深

        Args:
            raw_data: numpy 数组

        Returns:
            int: 推断的位深 (8, 10, 12, 14, 16)
        """
        max_val = int(np.max(raw_data))
        if max_val <= 255:
            return 8
        elif max_val <= 1023:
            return 10
        elif max_val <= 4095:
            return 12
        elif max_val <= 16383:
            return 14
        else:
            return 16


class ISPConfig:
    """ISP 管道的所有参数配置，带合理默认值"""

    def __init__(self):
        # ---------- 基础参数 ----------
        self.raw_w = 1920
        self.raw_h = 1080
        self.bayer_pattern = 'rggb'
        self.bit_depth = 10

        # ---------- DPC 参数 ----------
        self.dpc_thres = 30
        self.dpc_mode = 'gradient'
        self.dpc_clip = 1023

        # ---------- BLC 参数 ----------
        self.bl_r = 0
        self.bl_gr = 0
        self.bl_gb = 0
        self.bl_b = 0
        self.alpha = 0
        self.beta = 0
        self.blc_clip = 1023

        # ---------- AWB 参数 ----------
        self.r_gain = 1.5
        self.gr_gain = 1.0
        self.gb_gain = 1.0
        self.b_gain = 1.1
        self.awb_clip = 1023

        # ---------- CFA 参数 ----------
        self.cfa_mode = 'malvar'
        self.cfa_clip = 1023

        # ---------- CCM 参数 (3x4 矩阵，定点 Q10 格式) ----------
        self.ccm = np.array([
            [1024, 0, 0, 0],
            [0, 1024, 0, 0],
            [0, 0, 1024, 0]
        ], dtype=np.float64)

        # ---------- CSC 参数 (BT.601 RGB→YUV，已乘 1024) ----------
        self.csc = np.array([
            [0.257, 0.504, 0.098, 16],
            [-0.148, -0.291, 0.439, 128],
            [0.439, -0.368, -0.071, 128]
        ], dtype=np.float64)
        # CSC 系数乘以 1024（定点化）
        self.csc[:, 0:3] *= 1024
        self.csc[:, 3] *= 1024

        # ---------- BNF 参数 ----------
        self.bnf_dw = np.array([
            [8,  12,  32,  12,  8],
            [12, 64, 128,  64, 12],
            [32, 128, 1024, 128, 32],
            [12, 64, 128,  64, 12],
            [8,  12,  32,  12,  8]
        ], dtype=np.float64)
        self.bnf_rw = [0, 8, 16, 32]
        self.bnf_rthres = [128, 32, 8]
        self.bnf_clip = 255

        # ---------- Edge Enhancement 参数 ----------
        self.edge_filter = np.array([
            [-1, 0, -1, 0, -1],
            [-1, 0,  8, 0, -1],
            [-1, 0, -1, 0, -1]
        ], dtype=np.float64)
        self.ee_gain = [32, 128]
        self.ee_thres = [32, 64]
        self.ee_emclip = [-64, 64]

        # ---------- FCS 参数 ----------
        self.fcs_edge = [32, 64]
        self.fcs_gain = 32
        self.fcs_intercept = 2
        self.fcs_slope = 3

        # ---------- NLM 参数 ----------
        self.nlm_h = 15
        self.nlm_clip = 255

        # ---------- HSC 参数 ----------
        self.hue = 128
        self.saturation = 256
        self.hsc_clip = 255

        # ---------- BCC 参数 ----------
        self.brightness = 10
        self.contrast = 10
        self.bcc_clip = 255

        # ---------- Gamma 参数 ----------
        self.gamma = 0.5

    def update_clip_from_bits(self):
        """根据位深自动更新 clip 值"""
        bayer_clip = pow(2, self.bit_depth) - 1
        self.dpc_clip = bayer_clip
        self.blc_clip = bayer_clip
        self.awb_clip = bayer_clip
        self.cfa_clip = bayer_clip

    def load_from_csv(self, csv_path):
        """从 CSV 配置文件加载参数（覆盖默认值）"""
        if not os.path.exists(csv_path):
            print(f"[警告] 配置文件 {csv_path} 不存在，使用默认值")
            return

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) < 2:
                    continue
                param = row[0].strip()
                value = row[1].strip()
                self._set_param(param, value)

    def _set_param(self, param, value):
        """根据参数名设置对应的配置值"""
        param_map = {
            'raw_w': ('raw_w', int),
            'raw_h': ('raw_h', int),
            'dpc_thres': ('dpc_thres', int),
            'dpc_mode': ('dpc_mode', str),
            'dpc_clip': ('dpc_clip', int),
            'bayer_pattern': ('bayer_pattern', str),
            'bl_r': ('bl_r', int),
            'bl_gr': ('bl_gr', int),
            'bl_gb': ('bl_gb', int),
            'bl_b': ('bl_b', int),
            'alpha': ('alpha', int),
            'beta': ('beta', int),
            'blc_clip': ('blc_clip', int),
            'r_gain': ('r_gain', float),
            'gr_gain': ('gr_gain', float),
            'gb_gain': ('gb_gain', float),
            'b_gain': ('b_gain', float),
            'awb_clip': ('awb_clip', int),
            'cfa_mode': ('cfa_mode', str),
            'cfa_clip': ('cfa_clip', int),
            'bnf_clip': ('bnf_clip', int),
            'nlm_h': ('nlm_h', int),
            'nlm_clip': ('nlm_clip', int),
            'hue': ('hue', int),
            'saturation': ('saturation', int),
            'hsc_clip': ('hsc_clip', int),
            'brightness': ('brightness', int),
            'contrast': ('contrast', int),
            'bcc_clip': ('bcc_clip', int),
            'fcs_gain': ('fcs_gain', int),
            'fcs_intercept': ('fcs_intercept', int),
            'fcs_slope': ('fcs_slope', int),
        }

        # 简单参数直接映射
        if param in param_map:
            attr, converter = param_map[param]
            setattr(self, attr, converter(value))
            return

        # 矩阵参数: ccm_XX, csc_XX
        if param.startswith('ccm_') and len(param) == 6:
            r, c = int(param[4]), int(param[5])
            self.ccm[r][c] = int(value)
        elif param.startswith('csc_') and len(param) == 6:
            r, c = int(param[4]), int(param[5])
            self.csc[r][c] = 1024 * float(value) if c < 3 else 1024 * float(value)
        # BNF 距离权重: bnf_dw_XX
        elif param.startswith('bnf_dw_') and len(param) == 9:
            r, c = int(param[7]), int(param[8])
            self.bnf_dw[r][c] = int(value)
        # BNF 范围权重: bnf_rw_X
        elif param.startswith('bnf_rw_') and len(param) == 8:
            idx = int(param[7])
            self.bnf_rw[idx] = int(value)
        # BNF 阈值: bnf_rthres_X
        elif param.startswith('bnf_rthres_') and len(param) == 12:
            idx = int(param[11])
            self.bnf_rthres[idx] = int(value)
        # Edge filter: edge_filter_XX
        elif param.startswith('edge_filter_') and len(param) == 14:
            r, c = int(param[12]), int(param[13])
            self.edge_filter[r][c] = int(value)
        # EE 参数
        elif param == 'ee_gain_min':
            self.ee_gain[0] = int(value)
        elif param == 'ee_gain_max':
            self.ee_gain[1] = int(value)
        elif param == 'ee_thres_min':
            self.ee_thres[0] = int(value)
        elif param == 'ee_thres_max':
            self.ee_thres[1] = int(value)
        elif param == 'ee_emclip_min':
            self.ee_emclip[0] = int(value)
        elif param == 'ee_emclip_max':
            self.ee_emclip[1] = int(value)
        # FCS edge 参数
        elif param == 'fcs_edge_min':
            self.fcs_edge[0] = int(value)
        elif param == 'fcs_edge_max':
            self.fcs_edge[1] = int(value)


class ISPPipeline:
    """ISP 处理管道"""

    def __init__(self, config):
        """
        Args:
            config: ISPConfig 实例
        """
        self.cfg = config

    def run(self, rawimg):
        """执行完整 ISP 管道，从 RAW 到 RGB

        Args:
            rawimg: uint16 二维数组 (height, width)

        Returns:
            numpy.ndarray: uint8 RGB 图像 (height, width, 3)
        """
        cfg = self.cfg
        t0 = time.time()

        # 1. Dead Pixel Correction
        t = time.time()
        dpc = DPC(rawimg, cfg.dpc_thres, cfg.dpc_mode, cfg.dpc_clip)
        rawimg = dpc.execute()
        print(f"[1/15] 坏点校正 (DPC) 完成  {time.time()-t:.2f}s")

        # 2. Black Level Compensation
        t = time.time()
        bl_param = [cfg.bl_r, cfg.bl_gr, cfg.bl_gb, cfg.bl_b, cfg.alpha, cfg.beta]
        blc = BLC(rawimg, bl_param, cfg.bayer_pattern, cfg.blc_clip)
        rawimg = blc.execute()
        print(f"[2/15] 黑电平补偿 (BLC) 完成  {time.time()-t:.2f}s")

        # 3. Anti-aliasing Filter
        t = time.time()
        aaf = AAF(rawimg)
        rawimg = aaf.execute()
        print(f"[3/15] 抗锯齿滤波 (AAF) 完成  {time.time()-t:.2f}s")

        # 4. White Balance Gain Control
        t = time.time()
        awb_param = [cfg.r_gain, cfg.gr_gain, cfg.gb_gain, cfg.b_gain]
        awb = WBGC(rawimg, awb_param, cfg.bayer_pattern, cfg.awb_clip)
        rawimg = awb.execute()
        print(f"[4/15] 白平衡增益 (AWB) 完成  {time.time()-t:.2f}s")

        # 5. Chroma Noise Filtering
        t = time.time()
        cnf = CNF(rawimg, cfg.bayer_pattern, 0, awb_param, cfg.awb_clip)
        rawimg = cnf.execute()
        print(f"[5/15] 色度噪声滤波 (CNF) 完成  {time.time()-t:.2f}s")

        # 6. Color Filter Array Interpolation (Demosaicing)
        t = time.time()
        cfa = CFA(rawimg, cfg.cfa_mode, cfg.bayer_pattern, cfg.cfa_clip)
        rgbimg = cfa.execute()
        print(f"[6/15] 去马赛克 (CFA) 完成  {time.time()-t:.2f}s")

        # 7. Color Correction Matrix
        t = time.time()
        ccm_obj = CCM(rgbimg, cfg.ccm)
        rgbimg = ccm_obj.execute()
        print(f"[7/15] 颜色校正 (CCM) 完成  {time.time()-t:.2f}s")

        # 8. Gamma Correction
        t = time.time()
        bw = cfg.bit_depth
        maxval = pow(2, bw)
        ind = range(0, maxval)
        val = [round(pow(float(i) / maxval, cfg.gamma) * maxval) for i in ind]
        gamma_lut = dict(zip(ind, val))
        gc = GC(rgbimg, gamma_lut, 'rgb')
        rgbimg = gc.execute()
        print(f"[8/15] Gamma 校正 (GC) 完成  {time.time()-t:.2f}s")

        # 9. Color Space Conversion (RGB → YUV)
        t = time.time()
        csc_obj = CSC(rgbimg, cfg.csc)
        yuvimg = csc_obj.execute()
        print(f"[9/15] 色彩空间转换 (CSC) 完成  {time.time()-t:.2f}s")

        # 10. Non-Local Means Denoising (Y channel)
        t = time.time()
        nlm_obj = NLM(yuvimg[:, :, 0], 1, 4, cfg.nlm_h, cfg.nlm_clip)
        yuvimg_y = nlm_obj.execute()
        print(f"[10/15] 非局部均值去噪 (NLM) 完成  {time.time()-t:.2f}s")

        # 11. Bilateral Noise Filtering (Y channel)
        t = time.time()
        bnf = BNF(yuvimg_y, cfg.bnf_dw, cfg.bnf_rw, cfg.bnf_rthres, cfg.bnf_clip)
        yuvimg_y = bnf.execute()
        print(f"[11/15] 双边滤波 (BNF) 完成  {time.time()-t:.2f}s")

        # 12. Edge Enhancement (Y channel)
        t = time.time()
        ee = EE(yuvimg_y, cfg.edge_filter, cfg.ee_gain, cfg.ee_thres, cfg.ee_emclip)
        yuvimg_y, edgemap = ee.execute()
        print(f"[12/15] 边缘增强 (EE) 完成  {time.time()-t:.2f}s")

        # 13. False Color Suppression (UV channels)
        t = time.time()
        fcs_obj = FCS(yuvimg[:, :, 1:3], edgemap,
                      cfg.fcs_edge, cfg.fcs_gain, cfg.fcs_intercept, cfg.fcs_slope)
        yuvimg_uv = fcs_obj.execute()
        print(f"[13/15] 伪色抑制 (FCS) 完成  {time.time()-t:.2f}s")

        # 14. Hue/Saturation Control (UV channels)
        t = time.time()
        hsc = HSC(yuvimg_uv, cfg.hue, cfg.saturation, cfg.hsc_clip)
        yuvimg_uv = hsc.execute()
        print(f"[14/15] 色调/饱和度控制 (HSC) 完成  {time.time()-t:.2f}s")

        # 15. Brightness/Contrast Control (Y channel)
        t = time.time()
        contrast_val = cfg.contrast / pow(2, 5)
        bcc = BCC(yuvimg_y, cfg.brightness, contrast_val, cfg.bcc_clip)
        yuvimg_y = bcc.execute()
        print(f"[15/15] 亮度/对比度控制 (BCC) 完成  {time.time()-t:.2f}s")

        # 合成 YUV 输出
        raw_h, raw_w = cfg.raw_h, cfg.raw_w
        yuvimg_out = np.empty((raw_h, raw_w, 3), dtype=np.uint8)
        yuvimg_out[:, :, 0] = np.clip(yuvimg_y, 0, 255).astype(np.uint8)
        yuvimg_out[:, :, 1:3] = np.clip(yuvimg_uv, 0, 255).astype(np.uint8)

        # YUV → RGB 转换用于最终输出
        rgbimg_out = self._yuv2rgb(yuvimg_out)

        total = time.time() - t0
        print(f"\n{'='*50}")
        print(f"ISP 管道处理完成！总耗时: {total:.2f}s")
        print(f"输入: {raw_w}x{raw_h} RAW ({cfg.bit_depth}bit, {cfg.bayer_pattern})")
        print(f"输出: {raw_w}x{raw_h} RGB (8bit)")

        return rgbimg_out

    @staticmethod
    def _yuv2rgb(yuvimg):
        """YUV (BT.601) → RGB 转换

        Args:
            yuvimg: uint8 YUV 图像 (H, W, 3)

        Returns:
            numpy.ndarray: uint8 RGB 图像 (H, W, 3)
        """
        yuv = yuvimg.astype(np.float64)
        y = yuv[:, :, 0] - 16
        u = yuv[:, :, 1] - 128
        v = yuv[:, :, 2] - 128

        rgb = np.empty_like(yuvimg, dtype=np.float64)
        rgb[:, :, 0] = 1.164 * y + 1.596 * v                # R
        rgb[:, :, 1] = 1.164 * y - 0.392 * u - 0.813 * v    # G
        rgb[:, :, 2] = 1.164 * y + 2.017 * u                # B

        return np.clip(rgb, 0, 255).astype(np.uint8)


def load_raw_file(raw_path, width=None, height=None, bits=None):
    """加载 RAW 文件，支持自动推断参数

    Args:
        raw_path: RAW 文件路径
        width: 图像宽度（None 则自动推断）
        height: 图像高度（None 则自动推断）
        bits: 位深（None 则自动推断）

    Returns:
        tuple: (raw_data, width, height, bit_depth)
    """
    file_size = os.path.getsize(raw_path)
    analyzer = RawFileAnalyzer()

    # 自动推断分辨率
    if width is None or height is None:
        candidates = analyzer.guess_resolution(file_size)

        if not candidates:
            print(f"[错误] 无法从文件大小 ({file_size} 字节) 推断分辨率")
            print("请使用 -w 和 -h 参数手动指定宽度和高度")
            sys.exit(1)

        if len(candidates) == 1:
            width, height, bpp = candidates[0]
            print(f"[自动推断] 分辨率: {width}x{height} (每像素{bpp}字节)")
        else:
            print(f"[自动推断] 发现多个候选分辨率:")
            for i, (w, h, bpp) in enumerate(candidates):
                print(f"  [{i}] {w}x{h} (每像素{bpp}字节)")

            # 默认选择第一个（通常是 uint16 的候选）
            width, height, bpp = candidates[0]
            print(f"[自动选择] {width}x{height} (每像素{bpp}字节)")
            print("  如需其他分辨率，请使用 -w 和 -h 参数手动指定")
    else:
        # 用户指定了宽高，推断字节数
        total_pixels = width * height
        if file_size == total_pixels * 2:
            bpp = 2
        elif file_size == total_pixels:
            bpp = 1
        else:
            print(f"[错误] 文件大小 ({file_size}) 与指定的 {width}x{height} 不匹配")
            print(f"  期望: {total_pixels * 2} 字节 (uint16) 或 {total_pixels} 字节 (uint8)")
            sys.exit(1)

    # 读取 RAW 数据
    dtype = 'uint16' if bpp == 2 else 'uint8'
    raw_data = np.fromfile(raw_path, dtype=dtype)

    expected_pixels = width * height
    if raw_data.size < expected_pixels:
        print(f"[错误] 数据不足: 需要 {expected_pixels} 像素，文件只有 {raw_data.size} 像素")
        sys.exit(1)

    raw_data = raw_data[:expected_pixels].reshape((height, width))

    # 自动推断位深
    if bits is None:
        bits = analyzer.guess_bit_depth(raw_data)
        print(f"[自动推断] 位深: {bits} bit (数据最大值: {np.max(raw_data)})")
    else:
        print(f"[用户指定] 位深: {bits} bit")

    return raw_data, width, height, bits


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='openISP - RAW 图像到 RGB 图像处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python raw2rgb.py test.RAW                          # 自动推断所有参数
  python raw2rgb.py test.RAW -w 1920 -h 1080          # 指定宽高
  python raw2rgb.py test.RAW --bits 12 --bayer bggr   # 指定位深和 Bayer 模式
  python raw2rgb.py test.RAW -o output.png             # 指定输出文件
  python raw2rgb.py test.RAW --config config.csv       # 使用自定义配置
  python raw2rgb.py test.RAW --show                    # 处理后显示图像
        """)

    parser.add_argument('input', help='输入 RAW 文件路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出图像文件路径 (默认: <input>_rgb.png)')
    parser.add_argument('-w', '--width', type=int, default=None,
                        help='图像宽度 (自动推断)')
    parser.add_argument('-h', '--height', type=int, default=None,
                        help='图像高度 (自动推断)')
    parser.add_argument('--bits', type=int, default=None, choices=[8, 10, 12, 14, 16],
                        help='RAW 数据位深 (自动推断)')
    parser.add_argument('--bayer', default='rggb',
                        choices=['rggb', 'bggr', 'gbrg', 'grbg'],
                        help='Bayer 模式 (默认: rggb)')
    parser.add_argument('--config', default=None,
                        help='自定义 CSV 配置文件路径')
    parser.add_argument('--show', action='store_true',
                        help='处理后显示图像窗口')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Gamma 值 (默认: 0.5)')

    return parser


def main():
    parser = build_parser()

    # argparse 的 -h 与 --height 冲突，需要特殊处理
    # 将 -h 重新映射为帮助，使用 --height 作为高度参数
    # 重建 parser 以避免冲突
    parser = argparse.ArgumentParser(
        description='openISP - RAW 图像到 RGB 图像处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python raw2rgb.py test.RAW                                # 自动推断所有参数
  python raw2rgb.py test.RAW -W 1920 -H 1080                # 指定宽高
  python raw2rgb.py test.RAW --bits 12 --bayer bggr         # 指定位深和 Bayer
  python raw2rgb.py test.RAW -o output.png                   # 指定输出文件
  python raw2rgb.py test.RAW --config ./config/config.csv    # 使用自定义配置
  python raw2rgb.py test.RAW --show                          # 处理后显示图像
        """)

    parser.add_argument('input', help='输入 RAW 文件路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出图像文件路径 (默认: <input>_rgb.png)')
    parser.add_argument('-W', '--width', type=int, default=None,
                        help='图像宽度 (自动推断)')
    parser.add_argument('-H', '--height', type=int, default=None,
                        help='图像高度 (自动推断)')
    parser.add_argument('--bits', type=int, default=None, choices=[8, 10, 12, 14, 16],
                        help='RAW 数据位深 (自动推断)')
    parser.add_argument('--bayer', default='rggb',
                        choices=['rggb', 'bggr', 'gbrg', 'grbg'],
                        help='Bayer 模式 (默认: rggb)')
    parser.add_argument('--config', default=None,
                        help='自定义 CSV 配置文件路径')
    parser.add_argument('--show', action='store_true',
                        help='处理后显示图像窗口')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Gamma 值 (默认: 0.5)')

    args = parser.parse_args()

    # 验证输入文件
    if not os.path.exists(args.input):
        print(f"[错误] 输入文件不存在: {args.input}")
        sys.exit(1)

    print(f"{'='*50}")
    print(f"openISP - RAW 到 RGB 图像处理")
    print(f"{'='*50}")
    print(f"输入文件: {args.input}")
    print(f"文件大小: {os.path.getsize(args.input):,} 字节")
    print()

    # 1. 加载 RAW 文件（自动推断参数）
    rawimg, width, height, bit_depth = load_raw_file(
        args.input, args.width, args.height, args.bits
    )

    # 2. 创建配置
    cfg = ISPConfig()
    cfg.raw_w = width
    cfg.raw_h = height
    cfg.bit_depth = bit_depth
    cfg.bayer_pattern = args.bayer
    cfg.gamma = args.gamma
    cfg.update_clip_from_bits()

    # 如果提供了 CSV 配置，加载它（覆盖默认值）
    if args.config:
        print(f"[配置] 从 {args.config} 加载自定义参数")
        cfg.load_from_csv(args.config)

    print()

    # 3. 执行 ISP 管道
    pipeline = ISPPipeline(cfg)
    rgbimg = pipeline.run(rawimg)

    # 4. 保存输出
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + '_rgb.png'

    plt.imsave(args.output, rgbimg)
    print(f"\n[输出] RGB 图像已保存: {args.output}")

    # 5. 显示图像（可选）
    if args.show:
        plt.figure(figsize=(12, 8))
        plt.imshow(rgbimg)
        plt.title(f'openISP Output: {width}x{height}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
