# detect_armor
# YOLOv8s HQ200 训练程序使用说明

## 📄 程序信息

**文件**: `train_yolov8s_hq200.py`  
**功能**: 训练 YOLOv8-Small 装甲板检测模型  
**数据集**: HQ200 高质量数据集 (138训练图 + 40验证图)

## 🎯 程序作用

这个程序用于训练一个能够准确检测 RoboMaster 装甲板的深度学习模型。

训练出的模型可以：
- ✅ 识别红色装甲板
- ✅ 识别蓝色装甲板
- ✅ 识别机器人本体
- ✅ 在视频中实时检测

## 🚀 快速开始

### 1. 直接运行

```bash
# 方法1: 在项目根目录
python yolo/train_yolov8s_hq200.py

# 方法2: 在 yolo 目录
cd yolo
python train_yolov8s_hq200.py
```

### 2. 等待训练完成

训练需要 **3-5 小时** (CPU)，会显示进度：

```
==============================================================
  训练 YOLOv8s 装甲板检测模型
==============================================================

配置信息：
  - 基础模型: YOLOv8-Small (yolov8s.pt)
  - 数据集: HQ200 高质量数据集
  - 训练轮数: 50 epochs
  - 批次大小: 8
  - 图像尺寸: 640x640
  - 设备: CPU
==============================================================

[1/3] 加载 YOLOv8-Small 预训练模型...
✓ 模型加载成功

[2/3] 开始训练...
预计时间: 3-5 小时 (CPU)

Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
1/50      0G     1.844     3.31      1.985       85      640
2/50      0G     1.820     3.25      1.970       90      640
...
```

### 3. 训练完成

```
[3/3] 训练完成！
==============================================================

✓ 最佳模型保存在:
  runs/train/armor_hq200_v2/weights/best.pt

✓ 最后一轮模型:
  runs/train/armor_hq200_v2/weights/last.pt

✓ 训练结果图表:
  runs/train/armor_hq200_v2/results.png
```

## 📊 训练参数说明

### 基础配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | yolov8s.pt | YOLOv8-Small，比nano更大更准 |
| 训练轮数 | 50 | 充分学习特征 |
| 批次大小 | 8 | 每次处理8张图片 |
| 图像尺寸 | 640×640 | 标准尺寸 |
| 早停耐心 | 15 | 15轮无提升则停止 |

### 优化器配置

```python
optimizer="auto"      # 自动选择最佳优化器
lr0=0.01             # 初始学习率
lrf=0.01             # 最终学习率
momentum=0.937       # 动量
weight_decay=0.0005  # 权重衰减（L2正则）
```

### 数据增强配置

```python
# 颜色增强
hsv_h=0.015    # 色调抖动：小幅度，保持红蓝颜色稳定
hsv_s=0.7      # 饱和度：大幅度，适应不同光照
hsv_v=0.4      # 亮度：中等，模拟亮暗环境

# 几何变换
degrees=0.0     # 不旋转：装甲板有固定朝向
translate=0.1   # 平移：10%
scale=0.5       # 缩放：模拟远近距离
shear=0.0       # 不剪切
perspective=0.0 # 不透视变换

# 翻转
flipud=0.0  # 不垂直翻转：装甲板有上下
fliplr=0.5  # 水平翻转：50%概率

# 高级增强
mosaic=1.0              # Mosaic拼图：100%启用
mixup=0.0               # 不使用mixup
auto_augment="randaugment"  # 随机自动增强
erasing=0.4             # 随机擦除：40%
```

## 📈 训练监控

### 实时进度

训练过程中会每轮显示：

```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
3/50      0G     1.795     3.18      1.945       88      640
```

- **Epoch**: 当前轮数 / 总轮数
- **box_loss**: 边界框定位损失 ↓
- **cls_loss**: 分类损失 ↓
- **dfl_loss**: 分布焦点损失 ↓
- **Instances**: 检测到的实例数

**💡 提示**: 三个 loss 都会逐渐下降，如果不再下降说明已收敛

### 验证结果

每轮训练后会在验证集上测试：

```
               Class     Images  Instances      P      R  mAP50  mAP50-95
                 all         40        120  0.856  0.789  0.832     0.545
         armor_blue         40         45  0.883  0.822  0.871     0.592
          armor_red         40         50  0.845  0.780  0.815     0.521
               robot         40         25  0.840  0.765  0.810     0.523
```

- **P (Precision)**: 精确率，检测正确的比例
- **R (Recall)**: 召回率，找到目标的比例
- **mAP50**: IoU=0.5时的平均精度 ⭐ 主要指标
- **mAP50-95**: IoU=0.5-0.95的平均精度

## 📁 输出文件

训练完成后会生成以下文件：

```
runs/train/armor_hq200_v2/
├── weights/
│   ├── best.pt          ⭐ 最佳模型（用这个检测）
│   └── last.pt          最后一轮模型
├── results.png          训练曲线图
├── results.csv          详细数据
├── confusion_matrix.png 混淆矩阵
├── PR_curve.png         精确率-召回率曲线
├── F1_curve.png         F1分数曲线
├── labels.jpg           标签分布
└── args.yaml           训练参数记录
```

### 重要文件说明

#### 1. **best.pt** ⭐⭐⭐
最重要的文件！这是验证集上表现最好的模型。

使用方法：
```bash
python armor_detector_new.py \
  --source "视频.mp4" \
  --model "runs/train/armor_hq200_v2/weights/best.pt"
```

#### 2. **results.png**
训练曲线图，包含：
- 训练/验证损失曲线
- mAP曲线
- 精确率/召回率曲线

**如何查看**：双击打开图片

**如何判断训练好坏**：
- ✅ 好：曲线平滑下降并趋于稳定
- ❌ 差：曲线震荡或持续上升

#### 3. **confusion_matrix.png**
混淆矩阵，显示哪些类别容易混淆。

理想情况：对角线数字大（正确分类多），其他位置数字小（错误分类少）

## 🔧 高级使用

### 修改训练参数

编辑 `train_yolov8s_hq200.py` 文件：

#### 1. 增加训练轮数（提高精度）

```python
epochs=80,  # 从50改为80
```

#### 2. 减小批次（节省内存）

```python
batch=4,  # 从8改为4
```

#### 3. 调整学习率（微调）

```python
lr0=0.005,  # 降低学习率，训练更稳定
```

#### 4. 使用GPU加速（快10-50倍）

```python
device="0",  # 改为"0"使用第一块GPU
```

### 中断和恢复训练

训练可以随时中断（Ctrl+C），然后恢复：

```python
from ultralytics import YOLO

# 加载最后保存的模型
model = YOLO("runs/train/armor_hq200_v2/weights/last.pt")

# 继续训练
model.train(resume=True)
```

## 🎯 使用训练好的模型

### 检测视频

```bash
python armor_detector_new.py \
  --source "yolo/屏幕录制 2026-06-10 092602.mp4" \
  --model "runs/train/armor_hq200_v2/weights/best.pt" \
  --conf 0.20 \
  --output "result_v2.mp4"
```

### 检测图片

```bash
python armor_detector_new.py \
  --source "test/images" \
  --model "runs/train/armor_hq200_v2/weights/best.pt" \
  --output "results_v2"
```

### 批量测试

```bash
# 测试单张图片
python armor_detector_new.py \
  --source "test.jpg" \
  --model "runs/train/armor_hq200_v2/weights/best.pt"
```

## ❓ 常见问题

### Q1: 训练要多久？

**A**: 
- CPU: 3-5 小时
- GPU: 20-40 分钟

### Q2: 训练中断了怎么办？

**A**: 没关系！模型每轮都会保存。运行恢复代码：
```python
from ultralytics import YOLO
model = YOLO("runs/train/armor_hq200_v2/weights/last.pt")
model.train(resume=True)
```

### Q3: 内存不够怎么办？

**A**: 减小批次大小：
```python
batch=4,  # 或更小: batch=2
```

### Q4: 如何知道训练是否成功？

**A**: 检查以下指标：
1. 打开 `results.png`，loss曲线应该下降
2. mAP50 应该 > 0.6 (60%)
3. 用 best.pt 检测视频，看效果

### Q5: 训练完精度不高怎么办？

**A**: 尝试：
1. 增加训练轮数：`epochs=80`
2. 使用更大模型：`yolo11m.pt`
3. 增加训练数据
4. 调整数据增强参数

### Q6: 检测不到装甲板？

**A**: 
1. 确认使用的是 `best.pt` 不是 `last.pt`
2. 降低检测阈值：`--conf 0.15`
3. 检查训练是否收敛（查看 results.png）

### Q7: 红蓝颜色识别错误？

**A**: 这是正常的！训练脚本只训练**检测**装甲板位置。

颜色识别由 `armor_detector_new.py` 中的 HSV 颜色验证完成（已集成）。

### Q8: 可以训练其他数据集吗？

**A**: 可以！修改数据集路径：
```python
data="你的数据集.yaml",
```

## 📊 性能指标参考

### 良好的模型

- mAP50: > 0.60 (60%)
- mAP50-95: > 0.40 (40%)
- Precision: > 0.75 (75%)
- Recall: > 0.70 (70%)

### 优秀的模型

- mAP50: > 0.80 (80%)
- mAP50-95: > 0.55 (55%)
- Precision: > 0.85 (85%)
- Recall: > 0.80 (80%)

## 🔬 训练技巧

### 1. 查看训练进度

```bash
# 实时查看训练日志
tail -f runs/train/armor_hq200_v2/train.log
```

### 2. 对比不同模型

训练多个模型，对比效果：
```python
# 模型A: 50轮
epochs=50, name="model_50epochs"

# 模型B: 80轮
epochs=80, name="model_80epochs"
```

然后对比 `results.png` 和实际检测效果。

### 3. 监控资源使用

```bash
# CPU使用率
htop

# 内存使用
free -h
```

## 📞 获取帮助

### 查看详细日志

```bash
cat runs/train/armor_hq200_v2/train.log
```

### 查看训练参数

```bash
cat runs/train/armor_hq200_v2/args.yaml
```

### 重新训练

如果训练出问题，删除输出目录重新开始：
```bash
rm -rf runs/train/armor_hq200_v2
python train_yolov8s_hq200.py
```

## 🎓 下一步

1. ✅ 运行训练程序
2. ⏳ 等待 3-5 小时（可以在后台运行）
3. ✅ 查看 `results.png` 确认训练成功
4. ✅ 使用 `best.pt` 检测视频
5. ✅ 对比之前模型的效果

祝训练顺利！🚀

---

**相关文件**:
- 检测程序: `armor_detector_new.py`
- 完整训练指南: `../TRAINING_GUIDE.md`
- YOLOv11版本: `train_yolov11s_hq200.py`
