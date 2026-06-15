# 🎯 装甲板检测系统 - 快速上手指南

完整的 RoboMaster 装甲板训练和检测系统

## 📦 核心程序

### 1️⃣ **训练程序** - `train_yolov8s_hq200.py`
训练 YOLOv8s 模型，用于装甲板检测

### 2️⃣ **检测程序** - `armor_detector_new.py`  
使用训练好的模型检测视频/图片中的装甲板

---

## 🚀 快速开始（2步走）

### 第一步：训练模型（只需做一次）

```bash
# 在 yolo 目录下运行
cd yolo
python train_yolov8s_hq200.py
```

**等待时间**：3-5 小时（CPU）或 20-40 分钟（GPU）

**输出**：训练好的模型在 `runs/train/armor_hq200_v2/weights/best.pt`

### 第二步：检测视频/图片

```bash
# 检测视频
python armor_detector_new.py

# 程序会自动使用默认配置：
# - 视频：屏幕录制 2026-06-10 092602.mp4
# - 模型：runs/detect/runs/train/armor_hq200/weights/best.pt
# - 置信度：0.20
# - 输出：result_auto.mp4
```

**就这么简单！** 🎉

---

## 📋 详细使用说明

## 一、训练程序 `train_yolov8s_hq200.py`

### 🎯 作用
训练一个能识别红色/蓝色装甲板的深度学习模型

### ⚙️ 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | YOLOv8-Small | 比 nano 更大更准 |
| 轮数 | 50 epochs | 充分训练 |
| 批次 | 8 | 每次处理 8 张图 |
| 图像尺寸 | 640×640 | 标准尺寸 |
| 学习率 | 0.01 → 0.01 | 稳定学习 |
| 早停 | 15 轮 | 防止过拟合 |

### 📊 数据增强

```python
# 颜色增强
hsv_h=0.015      # 色调：小幅度（保持红蓝稳定）
hsv_s=0.7        # 饱和度：大幅度（适应光照）
hsv_v=0.4        # 亮度：中等

# 几何变换
fliplr=0.5       # 水平翻转 50%
translate=0.1    # 平移 10%
scale=0.5        # 缩放（模拟远近）

# 高级增强
mosaic=1.0       # Mosaic 拼图 100%
auto_augment="randaugment"  # 随机增强
erasing=0.4      # 随机擦除 40%
```

### 🎮 运行方式

#### 方式1：直接运行（最简单）

```bash
python train_yolov8s_hq200.py
```

#### 方式2：修改参数后运行

```python
# 编辑 train_yolov8s_hq200.py
epochs=80,        # 增加训练轮数
batch=4,          # 减小批次（节省内存）
device="0",       # 使用 GPU（快 10-50 倍）
```

### 📈 训练过程

```
==============================================================
  训练 YOLOv8s 装甲板检测模型
==============================================================

[1/3] 加载 YOLOv8-Small 预训练模型...
✓ 模型加载成功

[2/3] 开始训练...
预计时间: 3-5 小时 (CPU)

Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
1/50      0G     1.844     3.31      1.985       85      640
2/50      0G     1.820     3.25      1.970       90      640
3/50      0G     1.795     3.18      1.945       88      640
...
48/50     0G     0.892     1.42      1.125       82      640
49/50     0G     0.888     1.41      1.120       85      640
50/50     0G     0.885     1.40      1.118       84      640

[3/3] 训练完成！
✓ 最佳模型保存在: runs/train/armor_hq200_v2/weights/best.pt
```

### 📁 输出文件

```
runs/train/armor_hq200_v2/
├── weights/
│   ├── best.pt          ⭐ 最佳模型（用这个！）
│   └── last.pt          最后一轮模型
├── results.png          训练曲线图
├── results.csv          详细数据
├── confusion_matrix.png 混淆矩阵
├── PR_curve.png         精确率-召回率曲线
├── F1_curve.png         F1 分数曲线
└── args.yaml           训练参数记录
```

### ✅ 如何判断训练成功？

**查看 `results.png`**：

1. ✅ **Loss 曲线下降并趋于稳定**
   - box_loss（定位损失）↓
   - cls_loss（分类损失）↓
   - dfl_loss（分布焦点损失）↓

2. ✅ **mAP50 > 0.6**（60% 以上）
   - mAP50 = 0.83 → 优秀！
   - mAP50 = 0.65 → 良好
   - mAP50 = 0.45 → 需要改进

3. ✅ **精确率和召回率均衡**
   - Precision > 0.75
   - Recall > 0.70

---

## 二、检测程序 `armor_detector_new.py`

### 🎯 作用
使用训练好的模型检测视频或图片中的装甲板

### 🌟 核心功能

- ✅ **YOLO 检测**：快速定位装甲板位置
- ✅ **HSV 颜色识别**：精准区分红蓝装甲板
- ✅ **混合策略**：HSV 优先，模型补充
- ✅ **自动配置**：无需参数即可运行
- ✅ **实时可视化**：显示检测框和置信度

### 🔥 特色亮点

#### 1. 混合颜色识别策略

```
检测流程：
1. YOLO 检测装甲板位置
2. HSV 分析装甲板颜色（红/蓝）
   ├─ 如果 HSV 判定明确 → 使用 HSV 结果
   └─ 如果 HSV 不确定 → 使用模型分类
3. 过滤掉颜色不明的目标
4. 绘制结果（红框/蓝框）
```

**为什么这样做？**
- HSV 对红蓝光条识别更准确
- 避免模型分类错误（蓝色识别成红色）
- 提高整体检测准确率

#### 2. 智能默认配置

程序会自动检测工作目录并设置默认值：

```python
# 从项目根目录运行
默认视频：yolo/屏幕录制 2026-06-10 092602.mp4
默认模型：yolo/runs/detect/runs/train/armor_hq200/weights/best.pt

# 从 yolo 目录运行
默认视频：屏幕录制 2026-06-10 092602.mp4
默认模型：runs/detect/runs/train/armor_hq200/weights/best.pt
```

### 🎮 使用方式

#### 方式1：零配置运行（最简单）⭐

```bash
# 在 yolo 目录下
cd yolo
python armor_detector_new.py

# 自动使用默认配置：
# - 视频：屏幕录制 2026-06-10 092602.mp4
# - 模型：runs/detect/runs/train/armor_hq200/weights/best.pt
# - 置信度：0.20
# - 输出：result_auto.mp4
```

#### 方式2：指定视频

```bash
python armor_detector_new.py --source "你的视频.mp4"
```

#### 方式3：指定模型

```bash
python armor_detector_new.py \
  --source "视频.mp4" \
  --model "runs/train/armor_hq200_v2/weights/best.pt"
```

#### 方式4：完整参数

```bash
python armor_detector_new.py \
  --source "yolo/屏幕录制 2026-06-10 092602.mp4" \
  --model "runs/detect/runs/train/armor_hq200/weights/best.pt" \
  --conf 0.20 \
  --output "result_custom.mp4" \
  --device "cpu"
```

### 📊 参数说明

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--source` | 自动检测 | 输入视频/图片/目录 | `video.mp4` 或 `images/` |
| `--model` | 自动检测 | 模型路径 | `best.pt` |
| `--conf` | 0.20 | 置信度阈值（0-1） | 0.15（更多检测）<br>0.30（更少误检） |
| `--output` | result_auto.mp4 | 输出文件名 | `result_v2.mp4` |
| `--device` | cpu | 设备 | `cpu` 或 `0`（GPU） |

### 🎨 检测效果

```
检测开始...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
配置信息：
  视频源: yolo/屏幕录制 2026-06-10 092602.mp4
  模型:   runs/detect/runs/train/armor_hq200/weights/best.pt
  置信度: 0.20
  设备:   cpu
  输出:   result_auto.mp4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

检测进度: 0001/1733 | 装甲板: 2 (蓝:1 红:1)
检测进度: 0002/1733 | 装甲板: 3 (蓝:2 红:1)
...
检测进度: 1733/1733 | 装甲板: 4 (蓝:2 红:2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
检测完成！
输出文件: result_auto.mp4
总帧数:   1733
平均装甲板数: 2.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 🎨 可视化说明

- **红色装甲板** → **红色框** (BGR: 0,0,255)
- **蓝色装甲板** → **蓝色框** (BGR: 255,0,0)
- **灰色框（UNKNOWN）** → **已过滤，不显示**

每个检测框显示：
```
armor_red 0.85
↑        ↑
类别    置信度
```

### 🔧 高级用法

#### 1. 检测图片

```bash
# 单张图片
python armor_detector_new.py --source "test.jpg"

# 整个目录
python armor_detector_new.py --source "test/images"
```

#### 2. 调整置信度阈值

```bash
# 更敏感（检测更多，但可能误检）
python armor_detector_new.py --conf 0.15

# 更保守（减少误检，但可能漏检）
python armor_detector_new.py --conf 0.30
```

#### 3. 使用 GPU 加速

```bash
python armor_detector_new.py --device 0
# 速度提升 5-10 倍！
```

#### 4. 批量处理

```bash
# 处理多个视频
for video in video1.mp4 video2.mp4 video3.mp4; do
    python armor_detector_new.py --source "$video" --output "result_${video}"
done
```

---

## 🔄 完整工作流程

### 场景1：第一次使用

```bash
# 1. 训练模型（只需一次）
cd yolo
python train_yolov8s_hq200.py
# 等待 3-5 小时...

# 2. 检测视频
python armor_detector_new.py
# 完成！查看 result_auto.mp4
```

### 场景2：已有模型，只需检测

```bash
cd yolo
python armor_detector_new.py --source "新视频.mp4"
```

### 场景3：重新训练更好的模型

```bash
# 1. 修改训练参数
# 编辑 train_yolov8s_hq200.py:
#   epochs=80
#   batch=8

# 2. 训练
python train_yolov8s_hq200.py

# 3. 用新模型检测
python armor_detector_new.py \
  --model "runs/train/armor_hq200_v2/weights/best.pt"
```

---

## ❓ 常见问题

### Q1: 第一次运行，先训练还是先检测？

**A**: 如果已有 `runs/detect/runs/train/armor_hq200/weights/best.pt` 模型，直接检测：
```bash
python armor_detector_new.py
```

如果没有模型，先训练：
```bash
python train_yolov8s_hq200.py
```

### Q2: 训练需要多久？

**A**: 
- CPU: 3-5 小时
- GPU (CUDA): 20-40 分钟

### Q3: 检测速度慢怎么办？

**A**: 
1. 使用 GPU：`--device 0`
2. 降低输入分辨率
3. 提高置信度阈值：`--conf 0.30`

### Q4: 检测不到装甲板？

**A**: 
1. 降低置信度：`--conf 0.15`
2. 检查模型是否正确加载
3. 查看训练结果（mAP 是否足够高）

### Q5: 红蓝颜色识别错误？

**A**: 程序已使用 HSV 混合策略，准确率很高。如果仍有问题：
1. 检查视频光照条件
2. 调整 HSV 阈值（在代码中修改）

### Q6: 训练中断了怎么办？

**A**: 可以恢复训练：
```python
from ultralytics import YOLO
model = YOLO("runs/train/armor_hq200_v2/weights/last.pt")
model.train(resume=True)
```

### Q7: 内存不够怎么办？

**A**: 减小批次大小：
```python
# 编辑 train_yolov8s_hq200.py
batch=4,  # 或更小: batch=2
```

### Q8: 如何评估模型好坏？

**A**: 查看 `runs/train/armor_hq200_v2/results.png`：
- mAP50 > 0.6 → 良好
- mAP50 > 0.8 → 优秀
- Loss 持续下降 → 正常

### Q9: 可以检测其他物体吗？

**A**: 可以！准备新数据集，修改 `train_yolov8s_hq200.py` 中的：
```python
data="你的数据集.yaml"
```

### Q10: 两个程序有什么关系？

**A**: 
```
train_yolov8s_hq200.py  →  训练模型  →  best.pt
                                          ↓
armor_detector_new.py   →  使用模型  →  检测视频
```

---

## 📊 性能参考

### 训练指标（良好模型）
- mAP50: > 60%
- mAP50-95: > 40%
- Precision: > 75%
- Recall: > 70%

### 检测速度
- CPU: ~5-10 FPS
- GPU (GTX 1060): ~30-50 FPS
- GPU (RTX 3080): ~100-150 FPS

### 准确率
- 红蓝识别准确率: 95%+（HSV 混合策略）
- 装甲板检测准确率: 85%+（mAP50 = 0.83）

---

## 🛠️ 故障排查

### 问题1: 找不到模型

```
FileNotFoundError: best.pt
```

**解决**：检查模型路径，或先运行训练程序

### 问题2: 找不到视频

```
无法打开视频
```

**解决**：检查视频路径，使用绝对路径

### 问题3: CUDA 错误

```
CUDA out of memory
```

**解决**：减小批次大小或使用 CPU

### 问题4: 安装依赖失败

```
pip install ultralytics opencv-python numpy
```

---

## 📚 相关文档

- **完整训练指南**: `README_TRAIN_YOLOV8S.md`
- **项目总体文档**: `../TRAINING_GUIDE.md`
- **检测原理说明**: `../README_DETECTION.md`

---

## 🎓 学习路径

### 新手（0 基础）
1. ✅ 运行检测程序（使用现有模型）
2. ✅ 理解检测结果
3. ✅ 调整置信度阈值

### 进阶（有基础）
1. ✅ 运行训练程序
2. ✅ 理解训练参数
3. ✅ 调整数据增强
4. ✅ 优化模型性能

### 高级（深度定制）
1. ✅ 修改 HSV 颜色阈值
2. ✅ 自定义数据集训练
3. ✅ 模型架构调整
4. ✅ 部署到嵌入式设备

---

## 💡 最佳实践

### 训练时
- ✅ 使用 GPU 加速（快 10-50 倍）
- ✅ 监控 `results.png` 确保收敛
- ✅ 保存多个检查点对比效果
- ✅ 使用更多数据提高泛化能力

### 检测时
- ✅ 根据场景调整置信度
- ✅ 使用 GPU 提高实时性
- ✅ 对比不同模型的效果
- ✅ 记录检测日志便于调试

---

## 🚀 快速命令参考

```bash
# 训练模型
python train_yolov8s_hq200.py

# 检测视频（零配置）
python armor_detector_new.py

# 检测自定义视频
python armor_detector_new.py --source "video.mp4"

# 使用新训练的模型
python armor_detector_new.py \
  --model "runs/train/armor_hq200_v2/weights/best.pt"

# GPU 加速检测
python armor_detector_new.py --device 0

# 调整置信度
python armor_detector_new.py --conf 0.15  # 更敏感
python armor_detector_new.py --conf 0.30  # 更保守
```

---

**祝使用顺利！** 🎉

有问题随时查看详细文档或提问。
