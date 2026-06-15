"""
RoboMaster 装甲板红蓝识别 - 策略重构版
解决问题：
1. 修正黄色/黑色车体大面积背景色导致 HSV 误判红蓝的问题
2. 修复远端小车因灯条像素占比过低而被过滤、漏检的问题
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
# 鲁棒型 HSV 辅助颜色分析器
# ──────────────────────────────────────────────

class ArmorColorClassifier:
    """
    仅在模型置信度极低时作为辅助参考，优化了阈值范围以应对过曝
    """
    # 严格控制红色的 H 范围，隔绝车体黄色（20-30）
    RED_RANGES = [
        (np.array([0,   60,  90]),  np.array([8,   255, 255])),
        (np.array([172, 60,  90]),  np.array([180, 255, 255])),
    ]
    # 扩大蓝色 H 范围以包含青色，降低 S 容忍发白过曝
    BLUE_RANGE = (np.array([75, 45, 90]), np.array([135, 255, 255]))

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def classify(self, frame: np.ndarray, bbox: tuple) -> ArmorColor:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # 稍微向内收缩 5% 产生 ROI，尽量规避车辆外侧边缘的穿帮反光
        pad_w = int((x2 - x1) * 0.05)
        pad_h = int((y2 - y1) * 0.05)
        x1, y1 = max(0, x1 + pad_w), max(0, y1 + pad_h)
        x2, y2 = min(w, x2 - pad_w), min(h, y2 - pad_h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return ArmorColor.UNKNOWN

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 提取红色像素
        red_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for lo, hi in self.RED_RANGES:
            red_mask |= cv2.inRange(hsv, lo, hi)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self._kernel)
        red_cnt = cv2.countNonZero(red_mask)

        # 提取蓝色/青色像素
        blue_mask = cv2.inRange(hsv, *self.BLUE_RANGE)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self._kernel)
        blue_cnt = cv2.countNonZero(blue_mask)

        # 只有在某一种颜色具备绝对像素优势（比如是另一种的2倍以上）时才采信
        if red_cnt > blue_cnt * 2 and red_cnt > 5:
            return ArmorColor.RED
        elif blue_cnt > red_cnt * 2 and blue_cnt > 5:
            return ArmorColor.BLUE

        return ArmorColor.UNKNOWN


# ──────────────────────────────────────────────
# 主检测器
# ──────────────────────────────────────────────

class ArmorDetector:
    CLASS_NAMES = {0: "armor_blue", 1: "armor_red", 2: "robot"}
    DRAW_COLORS = {
        ArmorColor.RED:     (0,   0,   255),      # 红 -> BGR红
        ArmorColor.BLUE:    (255, 0,   0),        # 蓝 -> BGR蓝
        ArmorColor.UNKNOWN: (160, 160, 160),
    }

    def __init__(self, model_path: str, conf: float = 0.2, device: str = "cpu"):
        self.model = YOLO(model_path)
        # 将默认底线阈值降低到 0.2，确保能捕捉到远端的小车目标
        self.conf = conf
        self.device = device
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

                if class_id == 2:  # 过滤整车基座 robot 类别
                    final_color = ArmorColor.UNKNOWN
                    color_conf = 0.0
                elif class_id in (0, 1):
                    model_color = ArmorColor.BLUE if class_id == 0 else ArmorColor.RED
                    
                    # 💡 核心决策重构：
                    if conf > 0.45:
                        # 核心逻辑一：只要 YOLO 模型对形态分类足够自信（>0.45），直接无条件信任模型！
                        # 这样可以彻底避开大框里黄车车漆、黑车阴影对 HSV 的颜色干扰。
                        final_color = model_color
                        color_conf = conf
                    else:
                        # 核心逻辑二：只有在模型信心不足（低置信度、远端、或受干扰）时，才让 HSV 介入投票
                        hsv_color = self.color_clf.classify(frame, bbox)
                        if hsv_color != ArmorColor.UNKNOWN:
                            final_color = hsv_color
                            color_conf = 0.51  # 给予通过线以上的置信度
                        else:
                            # 如果 HSV 也无法给判断，继续维持模型本身的预判，绝不轻易抛弃目标
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
        
        # 核心逻辑三：允许显示高信度的目标，即使颜色被打上了 UNKNOWN，只要 class_id 正确也予以显示
        # 这样确保远端极其微小、数不出像素的小车不会在画面中“蒸发”
        for t in targets:
            if t.class_id == 2:  # 跳过 robot 框
                continue
                
            # 确定绘图颜色：若颜色明确则使用红/蓝，若未知则使用模型原始类别的颜色兜底
            if t.color != ArmorColor.UNKNOWN:
                color = self.DRAW_COLORS[t.color]
                color_name = t.color.value
            else:
                color = self.DRAW_COLORS[ArmorColor.BLUE] if t.class_id == 0 else self.DRAW_COLORS[ArmorColor.RED]
                color_name = "blue(M)" if t.class_id == 0 else "red(M)"

            x1, y1, x2, y2 = t.bbox
            w = x2 - x1

            # 画框
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # 四角强化线
            clen = max(6, w // 5)
            for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out, (px, py), (px + sx*clen, py), color, 2)
                cv2.line(out, (px, py), (px, py + sy*clen), color, 2)

            # 标签
            label = f"{color_name} | {t.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            bg_y = y1 - 6 if y1 > th + 8 else y2 + th + 6
            cv2.rectangle(out, (x1, bg_y - th - 4), (x1 + tw + 4, bg_y + 2), color, -1)
            cv2.putText(out, label, (x1 + 2, bg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.circle(out, t.center, 3, color, -1)

        return out

    def run_on_images(self, source: str, output_dir: str = "detect_results"):
        src  = Path(source)
        out  = Path(output_dir)
        out.mkdir(exist_ok=True)

        imgs = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) \
            if src.is_dir() else [src]

        print(f"共找到 {len(imgs)} 张图片。结果保存至: {out}/")
        win_name = "RoboMaster Armor Detection"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            targets = self.detect(frame)
            annotated = self.draw(frame, targets)

            cv2.imwrite(str(out / img_path.name), annotated)
            cv2.imshow(win_name, annotated)
            print(f"  [展示中]: {img_path.name} | 键盘输入任意键切换下一张...")
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def run_on_video(self, video_path: str, output_path: str = "output.mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        print(f"视频流处理中: {w}x{h} @ {fps:.1f}fps -> 输出到 {output_path}")
        win_name = "Real-time Video Detection"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            targets = self.detect(frame)
            annotated = self.draw(frame, targets)
            
            writer.write(annotated)
            cv2.imshow(win_name, annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断。")
                break
                
            if i % 30 == 0:
                print(f"  帧进度: {i:04d}/{total}")
            i += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print("处理完成。")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="RM装甲板红蓝分类器-重构版")
    
    if os.path.basename(os.getcwd()) == 'yolo':
        default_source = "屏幕录制 2026-06-10 092602.mp4"
        default_model = "runs\\detect\\runs\\train\\armor_hq200\\weights\\best.pt"
    else:
        default_source = "yolo\\屏幕录制 2026-06-10 092602.mp4"
        default_model = "yolo\\runs\\detect\\runs\\train\\armor_hq200\\weights\\best.pt"
    
    parser.add_argument("--source", default=default_source, help="输入源")
    parser.add_argument("--model",  default=default_model, help="模型权重")
    parser.add_argument("--conf",   type=float, default=0.20, help="YOLO基础置信度")
    parser.add_argument("--output", default="result_fixed.mp4", help="输出文件名")
    parser.add_argument("--device", default="cpu", help="推理设备")
    args = parser.parse_args()

    detector = ArmorDetector(args.model, args.conf, args.device)

    src = args.source.lower()
    if src.endswith((".mp4", ".avi", ".mov", ".mkv")):
        out = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
        detector.run_on_video(args.source, out)
    else:
        detector.run_on_images(args.source, args.output)


# """
# RoboMaster 装甲板红蓝识别 - 性能与鲁棒性终极优化版
# 策略：
# 1. 降低置信度底线，确保远端小目标（1、2、4、5号小车）绝不漏检。
# 2. 废除 YOLO 模型的红蓝分类，改用工业级通道差值（BGR Subtraction）对框内区域进行精准色彩裁决，彻底解决工程车大框误判、颜色闪烁问题。
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
#     confidence:       float   # YOLO位置置信度
#     color:            ArmorColor
#     center: tuple = field(init=False)

#     def __post_init__(self):
#         x1, y1, x2, y2 = self.bbox
#         self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


# # ──────────────────────────────────────────────
# # 工业级 BGR 通道差值颜色裁决器
# # ──────────────────────────────────────────────

# class ArmorColorDecider:
#     """
#     通过红蓝通道直接相减并结合亮度阈值，秒杀任何车体黄色噪点，且对远端微小发光点极度敏感
#     """
#     def __init__(self, diff_threshold: int = 15, brightness_floor: int = 70):
#         self.diff_threshold = diff_threshold      # 红蓝通道的最小有效差值
#         self.brightness_floor = brightness_floor  # 过滤暗部背景的亮度底线

#     def judge_color(self, frame: np.ndarray, bbox: tuple) -> ArmorColor:
#         x1, y1, x2, y2 = map(int, bbox)
#         h, w = frame.shape[:2]

#         # 【核心优化 1】向内收缩 15% 提取 ROI：
#         # 彻底切除 YOLO 大框边缘可能蹭到的黄色车漆、地面或者防护墙，只保留装甲板正中心！
#         box_w = x2 - x1
#         box_h = y2 - y1
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
#         crop_w = max(4, int(box_w * 0.35))
#         crop_h = max(4, int(box_h * 0.35))
        
#         sub_x1 = max(0, cx - crop_w)
#         sub_y1 = max(0, cy - crop_h)
#         sub_x2 = min(w, cx + crop_w)
#         sub_y2 = min(h, cy + crop_h)

#         roi = frame[sub_y1:sub_y2, sub_x1:sub_x2]
#         if roi.size == 0:
#             return ArmorColor.UNKNOWN

#         # 分离 BGR 通道
#         b_ch, g_ch, r_ch = cv2.split(roi)

#         # 【核心优化 2】通道相减法：
#         # 黄色车漆的 R 和 G 都很高，但 R - B 差值相对平衡。而红灯条的 R - B 极大，蓝灯条的 B - R 极大。
#         red_diff = cv2.subtract(r_ch, b_ch)
#         blue_diff = cv2.subtract(b_ch, r_ch)

#         # 结合亮度（灰度值）做掩膜，剔除环境中的黑色阴影或深色网格
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         bright_mask = gray > self.brightness_floor

#         # 统计有效发光区内的通道优势像素点数
#         red_pixels = np.sum((red_diff > self.diff_threshold) & bright_mask)
#         blue_pixels = np.sum((blue_diff > self.diff_threshold) & bright_mask)

#         # 谁具有压倒性优势，就判定为什么颜色
#         if red_pixels > blue_pixels and red_pixels > 2:
#             return ArmorColor.RED
#         elif blue_pixels > red_pixels and blue_pixels > 2:
#             return ArmorColor.BLUE
            
#         # 如果缩得太小或者极度暗淡（远端临界情况），做最后全局亮度兜底
#         sum_r = np.sum(r_ch.astype(np.float32) * bright_mask)
#         sum_b = np.sum(b_ch.astype(np.float32) * bright_mask)
#         if (sum_r + sum_b) > 0:
#             if sum_r > sum_b * 1.15: return ArmorColor.RED
#             if sum_b > sum_r * 1.15: return ArmorColor.BLUE

#         return ArmorColor.UNKNOWN


# # ──────────────────────────────────────────────
# # 主检测器
# # ──────────────────────────────────────────────

# class ArmorDetector:
#     DRAW_COLORS = {
#         ArmorColor.RED:     (0,   0,   255),  # 红色装甲板 → 红框
#         ArmorColor.BLUE:    (255, 0,   0),    # 蓝色装甲板 → 蓝框
#         ArmorColor.UNKNOWN: (0, 255, 255),    # 未知判定 → 黄框（方便调试看漏检）
#     }

#     def __init__(self, model_path: str, conf: float = 0.10, device: str = "cpu"):
#         self.model = YOLO(model_path)
#         # 【核心优化 3】底线置信度直接调到 0.10！
#         # 只要有一丝装甲板物理轮廓的可能性，就让 YOLO 框出来，后续交给颜色判定器过滤
#         self.conf = conf
#         self.device = device
#         self.color_decider = ArmorColorDecider()

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

#                 if class_id == 2: # 跳过车体 robot 大框
#                     continue

#                 # 无论 YOLO 模型说它是红是蓝，全部重新用通道亮度判定法剥离验证
#                 assigned_color = self.color_decider.judge_color(frame, bbox)

#                 targets.append(ArmorTarget(
#                     bbox=bbox,
#                     confidence=conf,
#                     color=assigned_color,
#                 ))
#         return targets

#     def draw(self, frame: np.ndarray, targets: list[ArmorTarget]) -> np.ndarray:
#         out = frame.copy()
        
#         # 为了应对全景图漏检，我们把判断为 UNKNOWN 的目标也画成黄色框，帮助你一眼看出是模型漏了还是颜色漏了
#         for t in targets:
#             x1, y1, x2, y2 = t.bbox
#             color = self.DRAW_COLORS[t.color]
#             w = x2 - x1

#             # 绘制框
#             cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

#             # 四角强化
#             clen = max(6, w // 5)
#             for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
#                 cv2.line(out, (px, py), (px + sx*clen, py), color, 2)
#                 cv2.line(out, (px, py), (px, py + sy*clen), color, 2)

#             # 标签：显示裁决出的颜色和位置置信度
#             label = f"{t.color.value} | {t.confidence:.2f}"
#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
#             bg_y = y1 - 6 if y1 > th + 8 else y2 + th + 6
#             cv2.rectangle(out, (x1, bg_y - th - 4), (x1 + tw + 4, bg_y + 2), color, -1)
#             cv2.putText(out, label, (x1 + 2, bg_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

#             cv2.circle(out, t.center, 3, color, -1)

#         # 顶部大字 HUD 计数
#         red  = sum(1 for t in targets if t.color == ArmorColor.RED)
#         blue = sum(1 for t in targets if t.color == ArmorColor.BLUE)
#         cv2.putText(out, f"RED:{red}  BLUE:{blue}  TOTAL:{len(targets)}",
#                     (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                     (0, 255, 255), 2, cv2.LINE_AA)
#         return out

#     def run_on_images(self, source: str, output_dir: str = "detect_results"):
#         src  = Path(source)
#         out  = Path(output_dir)
#         out.mkdir(exist_ok=True)

#         imgs = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) \
#             if src.is_dir() else [src]

#         print(f"共找到 {len(imgs)} 张图片。结果同步保存至: {out}/")
        
#         win_name = "RoboMaster Armor Detection"
#         cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#         for img_path in imgs:
#             frame = cv2.imread(str(img_path))
#             if frame is None:
#                 continue

#             targets = self.detect(frame)
#             annotated = self.draw(frame, targets)

#             cv2.imwrite(str(out / img_path.name), annotated)
#             cv2.imshow(win_name, annotated)
            
#             print(f"  [当前展示]: {img_path.name} | 窗口内按【任意键】看下一张...")
#             cv2.waitKey(0)

#         cv2.destroyAllWindows()

#     def run_on_video(self, video_path: str, output_path: str = "output.mp4"):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise FileNotFoundError(f"无法打开视频: {video_path}")

#         w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#         print(f"视频信息: {w}x{h} @ {fps:.1f}fps | 总计:{total}帧 -> {output_path}")
        
#         win_name = "Real-time Video Detection"
#         cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        
#         i = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             targets = self.detect(frame)
#             annotated = self.draw(frame, targets)
            
#             writer.write(annotated)
#             cv2.imshow(win_name, annotated)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("\n[提示] 用户按 'q' 键主动中止。")
#                 break
                
#             if i % 30 == 0:
#                 print(f"  帧进度: {i:04d}/{total}")
#             i += 1

#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()
#         print("处理完成。")


# # ──────────────────────────────────────────────
# # 入口
# # ──────────────────────────────────────────────

# if __name__ == "__main__":
#     import argparse
#     import os
    
#     parser = argparse.ArgumentParser(description="RM装甲板红蓝分类器-终极决策版")
    
#     if os.path.basename(os.getcwd()) == 'yolo':
#         default_source = "屏幕录制 2026-06-10 092602.mp4"
#         default_model = "runs\\detect\\runs\\train\\armor_hq200\\weights\\best.pt"
#     else:
#         default_source = "yolo\\屏幕录制 2026-06-10 092602.mp4"
#         default_model = "yolo\\runs\\detect\\runs\\train\\armor_hq200\\weights\\best.pt"
    
#     parser.add_argument("--source", default=default_source, help="输入源")
#     parser.add_argument("--model",  default=default_model, help="模型权重")
#     # 基础置信度放低到 0.10，全力活捉远端小车
#     parser.add_argument("--conf",   type=float, default=0.10, help="YOLO置信度底线")
#     parser.add_argument("--output", default="result_final.mp4", help="输出名")
#     parser.add_argument("--device", default="cpu", help="推理设备")
#     args = parser.parse_args()

#     detector = ArmorDetector(args.model, args.conf, args.device)

#     src = args.source.lower()
#     if src.endswith((".mp4", ".avi", ".mov", ".mkv")):
#         out = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
#         detector.run_on_video(args.source, out)
#     else:
#         detector.run_on_images(args.source, args.output)