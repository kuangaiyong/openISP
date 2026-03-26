# openISP 项目智能体约定

本文档定义了 openISP 项目的特定规范。所有内容继承自 `C:\Users\Administrator\.config\opencode\AGENTS.md` 的全局约定，**特别是中文回答规则**。

## 项目概述

- **项目名称**：Open Image Signal Processor (openISP)
- **类型**：纯 Python 图像信号处理 (ISP) 管道实现
- **技术栈**：Python 3.x + NumPy + Matplotlib + SciPy
- **核心依赖**：numpy、matplotlib、scipy（见 `requirements.txt`）
- **主要目录**：
  - `model/` - 15 个 ISP 算法模块（DPC、BLC、AWB、CFA、CCM 等）
  - `config/` - CSV 格式的配置文件
  - `raw/` - 测试用的 RAW 图像文件（10/12 位）
  - `docs/` - 文档和设计文件
- **关键工具模块**：
  - `model/blc.py` 导出 `BAYER_SLICES` 和 `get_bayer_slices()` — Bayer 模式统一查找表

## 构建和运行

### 主要脚本

| 命令 | 说明 |
|------|------|
| `python isp_pipeline.py` | 执行完整 ISP 管道，处理 test.RAW 图像 |
| `python raw2rgb.py test.RAW` | 简化入口：自动推断参数，一键处理 RAW→RGB |
| `python raw2rgb.py test.RAW -W 1920 -H 1080 --bits 12 --bayer rggb` | 手动指定参数 |
| `python test_bnf.py` | 测试双边滤波 (BNF) 模块，使用缩小尺寸图像 |

### 环境要求

- Python 3.6+
- numpy >= 1.15.0
- matplotlib >= 2.0.0

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行单个测试

```bash
# 运行特定算法的测试
python test_bnf.py

# 未来推荐使用 pytest（需先安装）
pip install pytest
pytest tests/ -v
pytest tests/test_algorithm.py::test_specific_function -v
```

## 代码风格扩展

### ISP 特定的命名约定

#### 变量名称
- **图像变量**：`rawimg`（RAW 格式）、`rgbimg`（RGB 格式）、`yuvimg`（YUV 格式）
- **参数名称**：模块简称 + 参数名，如 `dpc_thres`、`bnf_clip`、`cfa_mode`
- **配置值**：全小写下划线分隔，如 `raw_h`、`raw_w`、`bayer_pattern`

#### 类命名
- 单字母大写缩写 + 全名模式：`DPC`（Dead Pixel Correction）、`BLC`（Black Level Compensation）、`CFA`（Color Filter Array）

#### 关键常量
- **Bayer 模式**：`'rggb'`、`'bggr'`、`'gbrg'`、`'grbg'`
- **插值方法**：`'malvar'`、`'bilinear'` 等
- **处理模式**：`'mean'`、`'gradient'`

### 类结构规范

所有 ISP 算法模块遵循统一的类结构：

```python
class ALGORITHM_NAME:
    """算法简要描述"""
    
    def __init__(self, img, parameter1, parameter2, clip):
        self.img = img
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.clip = clip
    
    def execute(self):
        """执行算法处理，返回处理后的图像"""
        # 实现算法逻辑
        return self.clipping()
    
    def clipping(self):
        """裁剪输出值到有效范围"""
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img
```

### 关键注意事项

#### Bayer 模式处理
- 支持四种 Bayer 模式：rggb、bggr、gbrg、grbg
- 使用数组切片 `img[::2, ::2]` 提取特定颜色通道
- **必须验证输入的 Bayer 模式参数**

#### 数值精度
- 输入通常为 uint16（10-12 位 RAW 数据）
- 中间计算使用 int16/int32 避免溢出
- 最后用 `clipping()` 方法限制到有效范围
- **注意**：某些操作（如除法）需显式类型转换

#### 图像维度约定
- 内存布局：行优先（row-major）
- 访问方式：`img[y, x]` 或 `img[height, width]`
- 形状：`(height, width)` 或 `(height, width, channels)`
- **警告**：避免混淆高和宽的顺序

#### 配置文件格式
- 格式：CSV，字段为 `Variables, Values, Description`
- 路径：`./config/config.csv`（生产）、`./config/config_test.csv`（测试）
- 参数解析使用字符串匹配，如 `'dpc' in str(parameter)`

## 测试策略

### 当前测试方式
- `test_bnf.py` 是示例测试脚本
- 使用缩小尺寸的图像加快测试速度（128×72 vs 1280×720）
- 结果通过 matplotlib 可视化

### 推荐的 pytest 测试结构

```
tests/
├── test_dpc.py          # Dead Pixel Correction 测试
├── test_blc.py          # Black Level Compensation 测试
├── test_algorithm.py    # 通用算法测试
└── conftest.py          # pytest 配置和固件
```

### 单元测试示例

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块的测试
pytest tests/test_dpc.py -v

# 运行特定测试函数
pytest tests/test_dpc.py::test_dpc_mean_mode -v

# 显示打印输出
pytest tests/ -v -s
```

### 测试关键点
- 验证输出范围在 [0, clip_value] 内
- 检查 Bayer 模式切换的正确性
- 验证边界条件处理（图像边缘）
- 测试不同的输入尺寸和数据类型

## 错误处理扩展

### ISP 特定的验证

在所有模块的 `__init__` 方法中添加验证：

```python
def __init__(self, img, parameter, bayer_pattern, clip):
    # 验证图像类型
    assert img.ndim == 2, "输入图像应为 2D 数组"
    
    # 验证 Bayer 模式
    valid_patterns = {'rggb', 'bggr', 'gbrg', 'grbg'}
    assert bayer_pattern in valid_patterns, f"无效的 Bayer 模式: {bayer_pattern}"
    
    # 验证参数范围
    assert clip > 0, "clip 值应为正数"
    
    self.img = img
    self.parameter = parameter
    self.bayer_pattern = bayer_pattern
    self.clip = clip
```

## 代码示例与模式

### 处理 CSV 配置
```python
import csv

config_path = './config/config.csv'
with open(config_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        parameter = row[0]
        value = row[1]
        if 'dpc' in str(parameter):
            dpc_thres = int(value) if '_thres' in str(parameter) else dpc_thres
```

### NumPy 图像操作
```python
import numpy as np

# 创建空数组用于输出
output_img = np.empty((height, width), dtype=np.uint16)

# 按 Bayer 模式提取通道
r_channel = rawimg[::2, ::2]      # 红通道
gr_channel = rawimg[::2, 1::2]    # 绿(R)通道

# 填充和反射镜像
img_padded = np.pad(rawimg, (2, 2), 'reflect')

# 安全的数据类型转换
value_int = p0.astype(int)
result_uint16 = result_int.astype('uint16')
```

## 常见问题

### Q: 如何调试特定的算法模块？
A: 创建一个测试脚本，类似于 `test_bnf.py`，只加载所需的模块并输出中间结果。使用 matplotlib 可视化图像。

### Q: 如何处理不同的 RAW 图像大小？
A: 在配置文件中修改 `raw_w` 和 `raw_h`，确保 RAW 文件包含足够的数据。

### Q: 为什么输出图像看起来不对？
A: 检查：
1. Bayer 模式是否正确
2. 中间步骤的数据范围（是否溢出）
3. 图像尺寸和内存布局

---

**最后更新**：2026-03-26  
**版本**：1.0
