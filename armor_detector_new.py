
# """
# RoboMaster 装甲板红蓝识别
# YOLOv11检测 + 模型直接分类（不再使用HSV后处理）
# 用法：python armor_detector_new.py --source image.jpg --model best.pt

# 颜色方案：
# - 红色装甲板 → 红色框
# - 蓝色装甲板 → 蓝色框
# """

# import cv2
# import numpy as np
# from pathlib import Path
# from dataclasses import dataclass, field
# from enum import Enum
# from ultralytics import YOLO


# # ──────────────────────────────────────────────
# # 数据结构
# # ──────────────────────────────────────────────

# class ArmorColor(Enum):
#     RED     = "red"
#     BLUE    = "blue"
#     UNKNOWN = "unknown"


# @dataclass
# class ArmorTarget:
#     bbox:             tuple   # (x1, y1, x2, y2)
#     confidence:       float
#     class_id:         int
#     color:            ArmorColor
#     color_confidence: float
#     center: tuple = field(init=False)

#     def __post_init__(self):
#         x1, y1, x2, y2 = self.bbox
#         self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


# # ──────────────────────────────────────────────
# # HSV 颜色分类器
# # ──────────────────────────────────────────────

# class ArmorColorClassifier:
#     """
#     在装甲板 ROI 内通过 HSV 分割判断红/蓝灯条颜色
#     优化版：更严格的阈值，减少混淆
#     """

#     # 红色：更严格的饱和度和亮度要求
#     RED_RANGES = [
#         (np.array([0,   150, 100]),  np.array([10,  255, 255])),  # 低色调红
#         (np.array([170, 150, 100]),  np.array([180, 255, 255])),  # 高色调红
#     ]
#     # 蓝色：更窄的色调范围，更高的饱和度
#     BLUE_RANGE = (np.array([100, 150, 100]), np.array([124, 255, 255]))

#     def __init__(self, min_pixel_ratio: float = 0.02, min_confidence: float = 0.6):
#         """
#         min_pixel_ratio: 最小颜色像素占比（提高到 2%）
#         min_confidence: 最小置信度（需要 60% 以上才判定为该颜色）
#         """
#         self.min_pixel_ratio = min_pixel_ratio
#         self.min_confidence = min_confidence
#         self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#     def classify(self, frame: np.ndarray, bbox: tuple) -> tuple[ArmorColor, float]:
#         x1, y1, x2, y2 = map(int, bbox)
#         h, w = frame.shape[:2]

#         # 扩展 ROI 10%
#         pw = max(1, int((x2 - x1) * 0.1))
#         ph = max(1, int((y2 - y1) * 0.1))
#         x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
#         x2, y2 = min(w, x2 + pw), min(h, y2 + ph)

#         roi = frame[y1:y2, x1:x2]
#         if roi.size == 0:
#             return ArmorColor.UNKNOWN, 0.0

#         hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         total = roi.shape[0] * roi.shape[1]

#         # 红色
#         red_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
#         for lo, hi in self.RED_RANGES:
#             red_mask |= cv2.inRange(hsv, lo, hi)
#         red_mask  = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self._kernel)
#         red_cnt   = cv2.countNonZero(red_mask)

#         # 蓝色
#         blue_mask = cv2.inRange(hsv, *self.BLUE_RANGE)
#         blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self._kernel)
#         blue_cnt  = cv2.countNonZero(blue_mask)

#         color_total = red_cnt + blue_cnt
        
#         # 颜色像素太少，判定为 UNKNOWN
#         if color_total / total < self.min_pixel_ratio:
#             return ArmorColor.UNKNOWN, 0.0

#         # 计算置信度
#         if red_cnt >= blue_cnt:
#             confidence = red_cnt / color_total
#             # 置信度不够，判定为 UNKNOWN（避免误判）
#             if confidence < self.min_confidence:
#                 return ArmorColor.UNKNOWN, 0.0
#             return ArmorColor.RED, confidence
#         else:
#             confidence = blue_cnt / color_total
#             if confidence < self.min_confidence:
#                 return ArmorColor.UNKNOWN, 0.0
#             return ArmorColor.BLUE, confidence


# # ──────────────────────────────────────────────
# # 主检测器（图片模式）
# # ──────────────────────────────────────────────

# class ArmorDetector:
#     CLASS_NAMES = {0: "armor_blue", 1: "armor_red", 2: "robot"}
#     DRAW_COLORS = {
#         ArmorColor.RED:     (0,   0,   255),      # 红装甲板 → 红框 (BGR)
#         ArmorColor.BLUE:    (255, 0,   0),        # 蓝装甲板 → 蓝框 (BGR)
#         ArmorColor.UNKNOWN: (160, 160, 160),
#     }

#     def __init__(self, model_path: str, conf: float = 0.3, device: str = "cpu", debug: bool = False):
#         self.model    = YOLO(model_path)
#         self.conf     = conf
#         self.device   = device
#         self.debug    = debug  # 调试模式：显示 HSV 分割结果
#         self.color_clf = ArmorColorClassifier()

#     def detect(self, frame: np.ndarray) -> list[ArmorTarget]:
#         results = self.model.predict(
#             frame,
#             conf=self.conf,
#             device=self.device,
#             verbose=False,
#         )
#         targets = []
#         for result in results:
#             if result.boxes is None:
#                 continue
#             for box in result.boxes:
#                 bbox     = tuple(map(int, box.xyxy[0].tolist()))
#                 conf     = float(box.conf[0])
#                 class_id = int(box.cls[0])

#                 # 使用混合策略：模型+HSV双重验证
#                 if class_id == 2:  # robot类别
#                     final_color = ArmorColor.UNKNOWN
#                     color_conf = 0.0
#                 elif class_id in (0, 1):
#                     # 先用HSV验证颜色
#                     hsv_color, hsv_conf = self.color_clf.classify(frame, bbox)
#                     model_color = ArmorColor.BLUE if class_id == 0 else ArmorColor.RED
                    
#                     if hsv_color != ArmorColor.UNKNOWN:
#                         # HSV能识别，优先用HSV结果（更准确）
#                         final_color = hsv_color
#                         color_conf = hsv_conf
#                     else:
#                         # HSV无法识别，回退到模型结果
#                         final_color = model_color
#                         color_conf = conf
#                 else:
#                     final_color = ArmorColor.UNKNOWN
#                     color_conf = 0.0

#                 targets.append(ArmorTarget(
#                     bbox=bbox,
#                     confidence=conf,
#                     class_id=class_id,
#                     color=final_color,
#                     color_confidence=color_conf,
#                 ))
#         return targets

#     def draw(self, frame: np.ndarray, targets: list[ArmorTarget]) -> np.ndarray:
#         out = frame.copy()
#         # 只显示红色和蓝色装甲板，过滤掉 UNKNOWN（灰色框）
#         valid_targets = [t for t in targets if t.color != ArmorColor.UNKNOWN]
        
#         for t in valid_targets:
#             x1, y1, x2, y2 = t.bbox
#             color = self.DRAW_COLORS[t.color]
#             w = x2 - x1

#             # 矩形框
#             cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

#             # 四角标记
#             clen = max(8, w // 5)
#             for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
#                 cv2.line(out, (px, py), (px + sx*clen, py), color, 3)
#                 cv2.line(out, (px, py), (px, py + sy*clen), color, 3)

#             # 标签：显示模型分类结果
#             cls_name = self.CLASS_NAMES.get(t.class_id, "?")
#             label = f"{t.color.value} | {t.confidence:.2f}"
#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             bg_y = y1 - 6 if y1 > th + 8 else y2 + th + 6
#             cv2.rectangle(out, (x1, bg_y - th - 4), (x1 + tw + 4, bg_y + 2), color, -1)
#             cv2.putText(out, label, (x1 + 2, bg_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

#             cv2.circle(out, t.center, 4, color, -1)

#         # HUD - 只统计有效的红蓝装甲板
#         red  = sum(1 for t in valid_targets if t.color == ArmorColor.RED)
#         blue = sum(1 for t in valid_targets if t.color == ArmorColor.BLUE)
#         cv2.putText(out, f"RED:{red}  BLUE:{blue}  TOTAL:{len(valid_targets)}",
#                     (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                     (255, 255, 255), 2, cv2.LINE_AA)
#         return out

#     def run_on_images(self, source: str, output_dir: str = "detect_results"):
#         """对图片或图片目录做推理，结果保存到 output_dir"""
#         src  = Path(source)
#         out  = Path(output_dir)
#         out.mkdir(exist_ok=True)

#         imgs = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) \
#             if src.is_dir() else [src]

#         print(f"共 {len(imgs)} 张图片，结果保存至 {out}/")
#         for img_path in imgs:
#             frame = cv2.imread(str(img_path))
#             if frame is None:
#                 print(f"  跳过（无法读取）: {img_path.name}")
#                 continue

#             targets   = self.detect(frame)
#             # 过滤掉 UNKNOWN 颜色的目标
#             valid_targets = [t for t in targets if t.color != ArmorColor.UNKNOWN]
#             annotated = self.draw(frame, targets)

#             out_path = out / img_path.name
#             cv2.imwrite(str(out_path), annotated)

#             red  = sum(1 for t in valid_targets if t.color == ArmorColor.RED)
#             blue = sum(1 for t in valid_targets if t.color == ArmorColor.BLUE)
#             print(f"  {img_path.name}: 红{red} 蓝{blue} 共{len(valid_targets)}个目标")

#         print("完成")


#     def run_on_video(self, video_path: str, output_path: str = "output.mp4"):
#         """对视频做逐帧推理，结果保存为新视频"""
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise FileNotFoundError(f"无法打开视频: {video_path}")

#         w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         writer = cv2.VideoWriter(
#             output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
#         )

#         print(f"视频: {w}x{h} @ {fps:.1f}fps  共{total}帧 -> {output_path}")
#         i = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             targets   = self.detect(frame)
#             # 过滤掉 UNKNOWN 颜色的目标
#             valid_targets = [t for t in targets if t.color != ArmorColor.UNKNOWN]
#             annotated = self.draw(frame, targets)
#             writer.write(annotated)
#             if i % 30 == 0:
#                 red  = sum(1 for t in valid_targets if t.color == ArmorColor.RED)
#                 blue = sum(1 for t in valid_targets if t.color == ArmorColor.BLUE)
#                 print(f"  帧{i:04d}/{total}  红:{red} 蓝:{blue}")
#             i += 1

#         cap.release()
#         writer.release()
#         print(f"完成，共处理 {i} 帧 -> {output_path}")


# # ──────────────────────────────────────────────
# # 入口
# # ──────────────────────────────────────────────

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="装甲板检测")
#     parser.add_argument("--source", required=True, help="图片/图片目录/视频路径")
#     parser.add_argument("--model",  required=True, help="模型权重 .pt")
#     parser.add_argument("--conf",   type=float, default=0.3)
#     parser.add_argument("--output", default="output")
#     parser.add_argument("--device", default="cpu")
#     args = parser.parse_args()

#     detector = ArmorDetector(args.model, args.conf, args.device)

#     src = args.source.lower()
#     if src.endswith((".mp4", ".avi", ".mov", ".mkv")):
#         out = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
#         detector.run_on_video(args.source, out)
#     else:
#         detector.run_on_images(args.source, args.output)


"""
RoboMaster 装甲板红蓝识别
YOLOv11检测 + 模型直接分类 + 优化版 HSV 颜色双重验证（支持弹窗实时显示）
用法：python armor_detector_new.py --source image.jpg --model best.pt
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from ultralytics import YOLO


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

class ArmorColor(Enum):
    RED     = "red"
    BLUE    = "blue"
    UNKNOWN = "unknown"


@dataclass
class ArmorTarget:
    bbox:             tuple   # (x1, y1, x2, y2)
    confidence:       float
    class_id:         int
    color:            ArmorColor
    color_confidence: float
    center: tuple = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


# ──────────────────────────────────────────────
# 优化版 HSV 颜色分类器
# ──────────────────────────────────────────────

class ArmorColorClassifier:
    """
    在装甲板 ROI 内通过 HSV 分割判断红/蓝灯条颜色
    针对过曝发白、偏青绿色的灯条进行了鲁棒性调优
    """
    # 红色：降低饱和度(S)和亮度(V)阈值，防止过曝时无法识别
    RED_RANGES = [
        (np.array([0,   40,  80]),  np.array([10,  255, 255])),  # 低色调红
        (np.array([170, 40,  80]),  np.array([180, 255, 255])),  # 高色调红
    ]
    
    # 蓝色/青色：【重大修正】
    # 1. H下限由100下调至75，从而完美覆盖你图中那种偏“青色/翠绿色”的灯条
    # 2. S下限由150下调至40，容忍灯条中心因严重过曝而发白、饱和度变低的情况
    BLUE_RANGE = (np.array([75, 40, 80]), np.array([135, 255, 255]))

    def __init__(self, min_pixel_ratio: float = 0.01, min_confidence: float = 0.55):
        """
        min_pixel_ratio: 最小颜色像素占比（适当下调至 1%，防止框太大时被过滤）
        min_confidence: 最小置信度阈值
        """
        self.min_pixel_ratio = min_pixel_ratio
        self.min_confidence = min_confidence
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def classify(self, frame: np.ndarray, bbox: tuple) -> tuple[ArmorColor, float]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # 扩展 ROI 10%，确保包裹住边缘灯条
        pw = max(1, int((x2 - x1) * 0.1))
        ph = max(1, int((y2 - y1) * 0.1))
        x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
        x2, y2 = min(w, x2 + pw), min(h, y2 + ph)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return ArmorColor.UNKNOWN, 0.0

        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]

        # 红色掩膜
        red_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for lo, hi in self.RED_RANGES:
            red_mask |= cv2.inRange(hsv, lo, hi)
        red_mask  = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self._kernel)
        red_cnt   = cv2.countNonZero(red_mask)

        # 蓝色/青色掩膜
        blue_mask = cv2.inRange(hsv, *self.BLUE_RANGE)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self._kernel)
        blue_cnt  = cv2.countNonZero(blue_mask)

        color_total = red_cnt + blue_cnt
        
        # 如果ROI中红蓝像素实在太少，判定为 UNKNOWN
        if color_total / total < self.min_pixel_ratio:
            return ArmorColor.UNKNOWN, 0.0

        # 计算颜色置信度
        if red_cnt >= blue_cnt:
            confidence = red_cnt / color_total
            if confidence < self.min_confidence:
                return ArmorColor.UNKNOWN, 0.0
            return ArmorColor.RED, confidence
        else:
            confidence = blue_cnt / color_total
            if confidence < self.min_confidence:
                return ArmorColor.UNKNOWN, 0.0
            return ArmorColor.BLUE, confidence


# ──────────────────────────────────────────────
# 主检测器
# ──────────────────────────────────────────────

class ArmorDetector:
    CLASS_NAMES = {0: "armor_blue", 1: "armor_red", 2: "robot"}
    DRAW_COLORS = {
        ArmorColor.RED:     (0,   0,   255),      # 红色装甲板 → 红框
        ArmorColor.BLUE:    (255, 0,   0),        # 蓝色装甲板 → 蓝框
        ArmorColor.UNKNOWN: (160, 160, 160),
    }

    def __init__(self, model_path: str, conf: float = 0.3, device: str = "cpu"):
        self.model    = YOLO(model_path)
        self.conf     = conf
        self.device   = device
        self.color_clf = ArmorColorClassifier()

    def detect(self, frame: np.ndarray) -> list[ArmorTarget]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        targets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                bbox     = tuple(map(int, box.xyxy[0].tolist()))
                conf     = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id == 2:  # robot类别
                    final_color = ArmorColor.UNKNOWN
                    color_conf = 0.0
                elif class_id in (0, 1):
                    # 双重验证：先用改进的 HSV 提取颜色
                    hsv_color, hsv_conf = self.color_clf.classify(frame, bbox)
                    model_color = ArmorColor.BLUE if class_id == 0 else ArmorColor.RED
                    
                    if hsv_color != ArmorColor.UNKNOWN:
                        final_color = hsv_color
                        color_conf = hsv_conf
                    else:
                        # 如果HSV因为极端暴光没认出来，回退到YOLO模型的分类结果
                        final_color = model_color
                        color_conf = conf
                else:
                    final_color = ArmorColor.UNKNOWN
                    color_conf = 0.0

                targets.append(ArmorTarget(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    color=final_color,
                    color_confidence=color_conf,
                ))
        return targets

    def draw(self, frame: np.ndarray, targets: list[ArmorTarget]) -> np.ndarray:
        out = frame.copy()
        valid_targets = [t for t in targets if t.color != ArmorColor.UNKNOWN]
        
        for t in valid_targets:
            x1, y1, x2, y2 = t.bbox
            color = self.DRAW_COLORS[t.color]
            w = x2 - x1

            # 绘制装甲板核心框
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # 美化增强：绘制强化四角
            clen = max(8, w // 5)
            for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out, (px, py), (px + sx*clen, py), color, 3)
                cv2.line(out, (px, py), (px, py + sy*clen), color, 3)

            # 绘制标签信息
            label = f"{t.color.value} | {t.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bg_y = y1 - 6 if y1 > th + 8 else y2 + th + 6
            cv2.rectangle(out, (x1, bg_y - th - 4), (x1 + tw + 4, bg_y + 2), color, -1)
            cv2.putText(out, label, (x1 + 2, bg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 标出装甲板中心点
            cv2.circle(out, t.center, 4, color, -1)

        # 顶部大字 HUD 计数统计
        red  = sum(1 for t in valid_targets if t.color == ArmorColor.RED)
        blue = sum(1 for t in valid_targets if t.color == ArmorColor.BLUE)
        cv2.putText(out, f"RED:{red}  BLUE:{blue}  TOTAL:{len(valid_targets)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return out

    def run_on_images(self, source: str, output_dir: str = "detect_results"):
        """对图片进行推理，【已新增】实时弹窗窗口"""
        src  = Path(source)
        out  = Path(output_dir)
        out.mkdir(exist_ok=True)

        imgs = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) \
            if src.is_dir() else [src]

        print(f"共找到 {len(imgs)} 张图片。结果同步保存至: {out}/")
        
        # 创建一个可调大小的显示窗口
        win_name = "RoboMaster Armor Detection"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  跳过（无法读取）: {img_path.name}")
                continue

            targets = self.detect(frame)
            valid_targets = [t for t in targets if t.color != ArmorColor.UNKNOWN]
            annotated = self.draw(frame, targets)

            # 1. 保存到本地磁盘
            cv2.imwrite(str(out / img_path.name), annotated)

            # 2. 【核心新增】实时弹窗供用户查看
            cv2.imshow(win_name, annotated)
            
            red  = sum(1 for t in valid_targets if t.color == ArmorColor.RED)
            blue = sum(1 for t in valid_targets if t.color == ArmorColor.BLUE)
            print(f"  [当前展示]: {img_path.name} -> 红:{red} 蓝:{blue} | 窗口内按【任意键】看下一张...")
            
            # 阻塞等待按键
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        print("所有图片处理且浏览完毕。")


    def run_on_video(self, video_path: str, output_path: str = "output.mp4"):
        """对视频做逐帧推理，【已新增】实时画面播放"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        print(f"视频信息: {w}x{h} @ {fps:.1f}fps | 总计:{total}帧 -> 输出保存在:{output_path}")
        
        win_name = "Real-time Video Detection"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            targets = self.detect(frame)
            valid_targets = [t for t in targets if t.color != ArmorColor.UNKNOWN]
            annotated = self.draw(frame, targets)
            
            # 写入视频文件
            writer.write(annotated)
            
            # 【核心新增】实时刷新视频播放窗口
            cv2.imshow(win_name, annotated)
            
            # 监听键盘，按键盘上的 'q' 键可以随时强制退出视频播放
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[提示] 用户按 'q' 键主动中止了视频流播放。")
                break
                
            if i % 30 == 0:
                red  = sum(1 for t in valid_targets if t.color == ArmorColor.RED)
                blue = sum(1 for t in valid_targets if t.color == ArmorColor.BLUE)
                print(f"  正在播放第 {i:04d}/{total} 帧 | 画面中红装甲:{red} 蓝装甲:{blue}")
            i += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print("视频处理完成。")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="RM装甲板红蓝分类器升级版")
    
    # 自动检测工作目录，设置正确的默认路径
    if os.path.basename(os.getcwd()) == 'yolo':
        # 在 yolo 目录下运行
        default_source = "屏幕录制 2026-06-10 092602.mp4"
        default_model = "runs\\detect\\runs\\train\\armor_hq200\\weights\\best.pt"
    else:
        # 在项目根目录下运行
        default_source = "yolo\\屏幕录制 2026-06-10 092602.mp4"
        default_model = "yolo\\runs\\detect\\runs\\train\\armor_hq200\\weights\\best.pt"
    
    parser.add_argument("--source", 
                       default=default_source,
                       help="图片路径 / 图片目录 / 视频路径")
    parser.add_argument("--model",  
                       default=default_model,
                       help="YOLOv11 模型权重 (.pt)")
    parser.add_argument("--conf",   type=float, default=0.20, help="YOLO置信度阈值")
    parser.add_argument("--output", default="result_auto.mp4", help="输出文件夹或视频名")
    parser.add_argument("--device", default="cpu", help="推理设备: cpu 或 cuda")
    args = parser.parse_args()

    print("="*50)
    print("  RM装甲板检测程序")
    print("="*50)
    print(f"工作目录: {os.getcwd()}")
    print(f"输入: {args.source}")
    print(f"模型: {args.model}")
    print(f"置信度: {args.conf}")
    print(f"输出: {args.output}")
    print("="*50)
    print()

    detector = ArmorDetector(args.model, args.conf, args.device)

    src = args.source.lower()
    if src.endswith((".mp4", ".avi", ".mov", ".mkv")):
        out = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
        detector.run_on_video(args.source, out)
    else:
        detector.run_on_images(args.source, args.output)