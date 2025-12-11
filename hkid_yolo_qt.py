import sys
import os
from pathlib import Path
import re
import numpy as np
import cv2
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QComboBox,
    QTextEdit,
    QGroupBox,
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt

from ultralytics import YOLO
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

BASE_DIR = Path(__file__).resolve().parent
YOLO_MODELS = [
    {"id": 0, "label": "YOLOv11 Nano", "filename": "hkid_yolo_nano.pt"},
    {"id": 1, "label": "YOLOv11 Small", "filename": "hkid_yolo_small.pt"},
]
OCR_MODELS = [
    {"id": 0, "label": "PP-OCRv4 Server (高精度)", "ocr_version": "PP-OCRv4"},
    {"id": 1, "label": "PP-OCRv3 Mobile (輕量)", "ocr_version": "PP-OCRv3"},
]

CLASS_NAMES = {
    0: "card_type",
    1: "name_zh",
    2: "name_en",
    3: "dob",
    4: "sex",
    5: "first_issue_date",
    6: "issue_date",
    7: "id_number",
    8: "symbols",
}

FIELD_TITLES_ZH = {
    0: "證件類型",
    1: "中文姓名",
    2: "英文姓名",
    3: "出生日期",
    4: "性別",
    5: "首次簽發日期",
    6: "簽發日期",
    7: "身份證號碼",
    8: "符號標記",
}

OCR_LANG = "ch"

def to_halfwidth(s: str) -> str:
    result_chars = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            result_chars.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result_chars.append(chr(code - 0xFEE0))
        else:
            result_chars.append(ch)
    return "".join(result_chars)


def normalize_hkid(value: str) -> str:
    if not value:
        return value

    v = to_halfwidth(value)
    v = re.sub(r"\s+", "", v)
    core = re.sub(r"[()]", "", v)
    if len(core) < 2:
        return core

    prefix = core[:-1]
    check = core[-1]
    return f"{prefix}({check})"

def is_valid_hkid(value: str) -> bool:
    if not value:
        return False

    v = to_halfwidth(value).upper()
    s = re.sub(r"[^A-Z0-9]", "", v)

    m = re.match(r"^([A-Z]{1,2})(\d{6})([0-9A])$", s)
    if not m:
        return False

    prefix = m.group(1)
    digits = m.group(2)
    check_char = m.group(3)
    if len(prefix) == 1:
        chars = [" "] + list(prefix) + list(digits)
    else:
        chars = list(prefix) + list(digits)

    if len(chars) != 8:
        return False

    weights = [9, 8, 7, 6, 5, 4, 3, 2]

    def char_value(c: str) -> int:
        if c == " ":
            return 36
        if "A" <= c <= "Z":
            return ord(c) - ord("A") + 10
        if "0" <= c <= "9":
            return int(c)
        return 0

    total = sum(char_value(c) * w for c, w in zip(chars, weights))

    if check_char == "A":
        check_val = 10
    else:
        check_val = int(check_char)

    return (total + check_val) % 11 == 0

def normalize_chinese_name(value: str) -> str:
    if not value:
        return value
    v = to_halfwidth(value)
    v = re.sub(r"\s+", "", v)
    return v


def normalize_english_name(value: str) -> str:
    if not value:
        return value

    v = to_halfwidth(value)
    v = re.sub(r"\s+", " ", v).strip()
    parts = [p.strip() for p in v.split(",") if p.strip()]
    if not parts:
        return v

    surname = parts[0].upper()
    given = " ".join(parts[1:]).strip()

    if given:
        given = " ".join(w.capitalize() if w.isalpha() else w for w in given.split())
        return f"{surname}, {given}"
    else:
        return surname


def normalize_issue_code(value: str) -> str:
    if not value:
        return value

    v = to_halfwidth(value)
    v = re.sub(r"\s+", "", v)
    core = re.sub(r"[()]", "", v)

    if not core:
        return ""
    return f"({core})"


def normalize_symbols(value: str) -> str:
    if not value:
        return value

    v = to_halfwidth(value)
    v = re.sub(r"[\u4e00-\u9fff]+", "", v)
    v = re.sub(r"Date\s*of\s*[Il1]ssue", "", v, flags=re.IGNORECASE)
    v = re.sub(r"Date", "", v, flags=re.IGNORECASE)
    v = re.sub(r"Issue", "", v, flags=re.IGNORECASE)
    v = re.sub(r"\s+", "", v)
    v = v.upper()
    v = re.sub(r"[^A-Z*]", "", v)
    
    return v

def to_halfwidth(s: str) -> str:
    result_chars = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            result_chars.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result_chars.append(chr(code - 0xFEE0))
        else:
            result_chars.append(ch)
    return "".join(result_chars)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("香港新智能身份證 YOLO 識別程式（Local YOLO + OCR）")
        self.resize(1100, 650)

        self.yolo_model = None
        self.ocr = None
        self.current_image_path = None

        self.current_yolo_index = 0
        self.current_ocr_index = 0

        self._init_ui()
        self._load_yolo_model()
        self._load_ocr_model()
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)
        top_layout.addWidget(QLabel("YOLO 模型："))
        self.yolo_combo = QComboBox()
        for m in YOLO_MODELS:
            self.yolo_combo.addItem(m["label"], userData=m["id"])
        self.yolo_combo.setCurrentIndex(self.current_yolo_index)
        self.yolo_combo.currentIndexChanged.connect(self.on_yolo_model_changed)
        top_layout.addWidget(self.yolo_combo)
        top_layout.addWidget(QLabel("OCR 模型："))
        self.ocr_combo = QComboBox()
        for m in OCR_MODELS:
            self.ocr_combo.addItem(m["label"], userData=m["id"])
        self.ocr_combo.setCurrentIndex(self.current_ocr_index)
        self.ocr_combo.currentIndexChanged.connect(self.on_ocr_model_changed)
        top_layout.addWidget(self.ocr_combo)
        self.btn_select = QPushButton("選擇圖片")
        self.btn_select.clicked.connect(self.select_image)
        top_layout.addWidget(self.btn_select)

        self.btn_run = QPushButton("執行識別")
        self.btn_run.clicked.connect(self.run_detection)
        top_layout.addWidget(self.btn_run)
        self.status_label = QLabel("初始化中...")
        top_layout.addWidget(self.status_label, stretch=1)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)
        self.image_label = QLabel("尚未載入圖片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        content_layout.addWidget(self.image_label, stretch=1)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(right_widget, stretch=1)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["欄位", "Class ID", "Class Name", "Score", "OCR 結果"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(320)
        right_layout.addWidget(self.table)
        staff_group = QGroupBox("輸出資訊")
        staff_layout = QVBoxLayout()
        staff_group.setLayout(staff_layout)
        
        self.staff_info_text = QTextEdit()
        self.staff_info_text.setReadOnly(True)
        self.staff_info_text.setMinimumHeight(300)
        staff_layout.addWidget(self.staff_info_text)
        
        right_layout.addWidget(staff_group)
        right_layout.addStretch()
        bottom_group = QGroupBox("訊息顯示")
        bottom_layout = QVBoxLayout()
        bottom_group.setLayout(bottom_layout)

        self.bottom_info_text = QTextEdit()
        self.bottom_info_text.setReadOnly(True)
        self.bottom_info_text.setPlaceholderText("等待中...")
        self.bottom_info_text.setMaximumHeight(100)
        bottom_layout.addWidget(self.bottom_info_text)

        main_layout.addWidget(bottom_group)

    def _postprocess_field(self, cls_id: int, value: str) -> str:
        if not value:
            return value

        v = to_halfwidth(value)

        if cls_id == 7:
            return normalize_hkid(v)
        elif cls_id == 1:
            return normalize_chinese_name(v)
        elif cls_id == 2:
            return normalize_english_name(v)
        elif cls_id == 5:
            return normalize_issue_code(v)
        elif cls_id == 8:
            return normalize_symbols(v)
        else:
            return v
    def _get_current_yolo_path(self) -> Path:
        cfg = YOLO_MODELS[self.current_yolo_index]
        return BASE_DIR / "models" / "yolo" / cfg["filename"]

    def _load_yolo_model(self):
        model_path = self._get_current_yolo_path()

        if not model_path.exists():
            self.status_label.setText(f"找不到 YOLO 模型檔案：{model_path}")
            QMessageBox.critical(
                self,
                "錯誤",
                f"找不到 YOLO 模型檔案：\n{model_path}\n\n"
                "請確認 models/yolo/ 下面的檔名是否正確。",
            )
            self.yolo_model = None
            return

        try:
            self.yolo_model = YOLO(str(model_path))
            self.status_label.setText(f"YOLO 模型已載入：{model_path.name}")
        except Exception as e:
            self.yolo_model = None
            self.status_label.setText("YOLO 模型載入失敗")
            QMessageBox.critical(self, "錯誤", f"載入 YOLO 模型時發生錯誤：\n{e}")

    def on_yolo_model_changed(self, index: int):
        self.current_yolo_index = index
        self.status_label.setText("切換 YOLO 模型中...")
        self._load_yolo_model()
    def _get_current_ocr_config(self):
        return OCR_MODELS[self.current_ocr_index]

    def _load_ocr_model(self):
        cfg = self._get_current_ocr_config()
        try:
            self.ocr = PaddleOCR(
                lang=OCR_LANG,
                ocr_version=cfg["ocr_version"],
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
            )
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = list(self.ocr.predict(dummy))

            self.status_label.setText(
                self.status_label.text()
                + f" | OCR 模型已載入：{cfg['label']}"
            )
        except Exception as e:
            self.ocr = None
            self.status_label.setText("OCR 模型載入失敗")
            QMessageBox.critical(self, "錯誤", f"載入 OCR 模型時發生錯誤：\n{e}")

    def on_ocr_model_changed(self, index: int):
        self.current_ocr_index = index
        self.status_label.setText("切換 OCR 模型中...")
        self._load_ocr_model()
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self._show_image(file_path)
        self.status_label.setText(f"已選擇圖片：{os.path.basename(file_path)}")

    def _show_image(self, file_path: str):
        pixmap = QPixmap(file_path)
        self._set_image_label_pixmap(pixmap)

    def _set_image_label_pixmap(self, pixmap: QPixmap):
        if pixmap.isNull():
            self.image_label.setText("無法載入圖片")
            return

        label_size = self.image_label.size()
        if label_size.width() == 0 or label_size.height() == 0:
            scaled = pixmap.scaledToWidth(500, Qt.SmoothTransformation)
        else:
            scaled = pixmap.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self.image_label.setPixmap(scaled)
    def _run_ocr_on_crop(self, crop_np: np.ndarray) -> str:
        if self.ocr is None:
            return ""
        try:
            crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
        except Exception:
            crop_bgr = crop_np

        try:
            outputs = self.ocr.predict(crop_bgr)
        except Exception as e:
            print("OCR predict error:", e)
            return ""
        if not isinstance(outputs, (list, tuple)):
            try:
                outputs = list(outputs)
            except TypeError:
                outputs = [outputs]

        if not outputs:
            return ""

        first = outputs[0]
        texts = []
        if hasattr(first, "json"):
            try:
                j = first.json

                res = j.get("res", j)
                if isinstance(res, list):
                    for item in res:
                        if not isinstance(item, dict):
                            continue
                        t = item.get("text") or item.get("rec_text")
                        if t:
                            texts.append(str(t))
                elif isinstance(res, dict):
                    if "rec_text" in res:
                        texts.append(str(res["rec_text"]))
                    if "rec_texts" in res and isinstance(res["rec_texts"], (list, tuple)):
                        texts.extend(str(t) for t in res["rec_texts"])

            except Exception as e:
                print("Parse Result.json error:", e)
        elif isinstance(first, list):
            lines = first
            try:
                for item in lines:
                    if len(item) >= 2:
                        t = item[1][0]
                        texts.append(str(t))
            except Exception as e:
                print("Parse legacy OCR result error:", e)

        if not texts:
            return ""

        merged = " ".join(t.strip() for t in texts if t and str(t).strip())
        merged = to_halfwidth(merged)

        return merged
    def run_detection(self):
        if self.yolo_model is None:
            QMessageBox.warning(self, "提示", "YOLO 模型尚未載入成功。")
            return

        if self.ocr is None:
            QMessageBox.warning(self, "提示", "OCR 模型尚未載入成功。")
            return

        if not self.current_image_path:
            QMessageBox.information(self, "提示", "請先選擇一張圖片。")
            return

        try:
            self.status_label.setText("推理中（YOLO + OCR）...")
            QApplication.processEvents()
            results = self.yolo_model(self.current_image_path)[0]

            base_img = Image.open(self.current_image_path).convert("RGB")
            draw_img = base_img.copy()
            draw = ImageDraw.Draw(draw_img)

            img_w, img_h = base_img.size
            for box in results.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                x1, y1, x2, y2 = xyxy

                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

                label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                text_x = int(x1) + 2
                text_y = max(0, int(y1) - 18)
                bg_w = int(10 * len(label))
                draw.rectangle(
                    [(text_x, text_y), (text_x + bg_w, text_y + 18)],
                    fill=(255, 0, 0),
                )
                draw.text((text_x + 2, text_y + 2), label, fill=(255, 255, 255))

            draw_np = np.array(draw_img)
            h, w, ch = draw_np.shape
            bytes_per_line = ch * w
            q_img = QImage(draw_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
            rows = []
            dob_box = None

            for box in results.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                x1, y1, x2, y2 = xyxy
                
                if cls_id == 3:
                    dob_box = (x1, y1, x2, y2)

                x1_c = max(0, int(x1))
                y1_c = max(0, int(y1))
                x2_c = min(img_w, int(x2))
                y2_c = min(img_h, int(y2))

                text_value = ""

                if x2_c > x1_c and y2_c > y1_c:
                    w_box = x2_c - x1_c
                    h_box = y2_c - y1_c
                    if w_box >= 8 and h_box >= 8:
                        try:
                            crop = base_img.crop((x1_c, y1_c, x2_c, y2_c))
                            crop_np = np.array(crop)
                            text_value = self._run_ocr_on_crop(crop_np)
                            text_value = self._postprocess_field(cls_id, text_value)
                        except Exception as e:
                            print(f"OCR error on class {cls_id}: {e}")
                            text_value = ""

                rows.append(
                    {
                        "field_title": FIELD_TITLES_ZH.get(cls_id, ""),
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                        "score": score,
                        "value": text_value,
                    }
                )
            if dob_box:
                dx1, dy1, dx2, dy2 = dob_box
                w_dob = dx2 - dx1
                h_dob = dy2 - dy1
                
                sy1 = dy2
                sy2 = dy2 + h_dob * 1.5
                sx1 = dx1
                sx2 = dx2 + w_dob * 0.5

                sx1_c = max(0, int(sx1))
                sy1_c = max(0, int(sy1))
                sx2_c = min(img_w, int(sx2))
                sy2_c = min(img_h, int(sy2))

                if sx2_c > sx1_c and sy2_c > sy1_c:
                    try:
                        crop_sym = base_img.crop((sx1_c, sy1_c, sx2_c, sy2_c))
                        crop_sym_np = np.array(crop_sym)
                        
                        draw.rectangle([(sx1, sy1), (sx2, sy2)], outline=(255, 0, 255), width=2)
                        
                        sym_text = self._run_ocr_on_crop(crop_sym_np)
                        
                        if sym_text:
                            rows.append({
                                "field_title": FIELD_TITLES_ZH.get(8, "符號標記"),
                                "class_id": 8,
                                "class_name": "symbols",
                                "score": 1.0,
                                "value": sym_text
                            })
                    except Exception as e:
                        print(f"Symbol OCR error: {e}")

            draw_np = np.array(draw_img)
            h, w, ch = draw_np.shape
            bytes_per_line = ch * w
            q_img = QImage(draw_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._set_image_label_pixmap(QPixmap.fromImage(q_img))

            self._update_table(rows)
            self._update_staff_info(rows)
            self.status_label.setText(
                f"推理完成：{len(rows)} 個欄位（已包含 OCR 結果）。"
            )

        except Exception as e:
            self.status_label.setText("推理失敗")
            QMessageBox.critical(self, "錯誤", f"推理時發生錯誤：\n{e}")
    def _update_table(self, rows):
        self.table.setRowCount(len(rows))

        for row_idx, b in enumerate(rows):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(b["field_title"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(b["class_id"])))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(b["class_name"])))
            self.table.setItem(row_idx, 3, QTableWidgetItem(f"{b['score']:.3f}"))

            value_item = QTableWidgetItem(b["value"] or "")

            if b["class_id"] == 7 and b["value"]:
                if is_valid_hkid(b["value"]):
                    value_item.setForeground(QColor(0, 160, 0))
                else:
                    value_item.setForeground(QColor(200, 0, 0))

            self.table.setItem(row_idx, 4, value_item)

        self.table.resizeColumnsToContents()

    def _update_staff_info(self, rows):
        data_map = {row['class_id']: row['value'] for row in rows}
        symbols = data_map.get(8, "")
        c_type = data_map.get(0, "")
        res_status = "非永久"
        if symbols and ("***" in symbols or "A" in symbols):
            res_status = "永久"
        elif "永久" in c_type:
            res_status = "永久"
        elif not c_type and not symbols:
            res_status = ""
        id_age_type = "成人身份證"
        if res_status == "永久" and symbols.count('*') == 1:
            id_age_type = "兒童身份證"
        elif not res_status:
            id_age_type = ""
        res_code_map = {
            "A": "有香港居留權",
            "R": "有香港入境權",
            "C": "登記時在香港的居留受限",
            "U": "登記時在香港的居留不受限",
        }
        res_code_list = []
        for char, desc in res_code_map.items():
            if char in symbols:
                res_code_list.append(f"{char} - {desc}")
        res_code_display = "<br>".join(res_code_list)
        birth_place_map = {
            "Z": "香港",
            "X": "內地",
            "W": "澳門",
            "O": "其他地區",
        }
        birth_place_list = []
        for char, desc in birth_place_map.items():
            if char in symbols:
                birth_place_list.append(f"{char} - {desc}")
        birth_place_display = "<br>".join(birth_place_list)
        dob_str = data_map.get(3, "")
        age_info = ""
        calculated_age = None
        if dob_str:
            try:
                clean_dob = re.sub(r"[^0-9\-]", "", dob_str)
                dob_date = datetime.strptime(clean_dob, "%d-%m-%Y")
                today = datetime.now()
                age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                calculated_age = age
                
                if age >= 18:
                    age_info = f" <span style='color: green;'>(已滿18歲)</span>"
                else:
                    age_info = f" <span style='color: red;'>(未滿18歲)</span>"
            except Exception:
                pass
        display_order = [
            (1, "中文姓名\t\t"),
            (2, "英文姓名\t\t"),
            (7, "身份證號碼\t\t"),
            ("resident_status", "居民身份\t\t"),
            ("id_age_type", "證件類型\t\t"),
            ("residential_code", "居留身份\t\t"),
            ("birth_place", "申報出生地\t\t"),
            (3, "出生日期\t\t"),
            (4, "性別\t\t"),
            (6, "簽發日期\t\t"),
            (5, "首次簽發日期\t\t"),
        ]

        lines = []
        html_content = ""
        html_content += "<div style='font-size: 14pt; line-height: 1.5;'>"
        
        for cls_id, title in display_order:
            val_display = ""
            
            if cls_id == "resident_status":
                val_display = res_status
            elif cls_id == "id_age_type":
                val_display = id_age_type
            elif cls_id == "residential_code":
                val_display = res_code_display
            elif cls_id == "birth_place":
                val_display = birth_place_display
            elif cls_id == 3:
                val = data_map.get(cls_id, "")
                val_display = val + age_info
            else:
                val = data_map.get(cls_id, "")
                if cls_id == 7 and val:
                    if is_valid_hkid(val):
                        val_display = f"<span style='color: green;'>{val}</span>"
                    else:
                        val_display = f"<span style='color: red;'>{val} (無效)</span>"
                else:
                    val_display = val

            html_content += f"<b>{title}:</b> {val_display}<br>"
        
        html_content += "</div>"
        self.staff_info_text.setHtml(html_content)
        
        base_msg = "請核對上述資料是否與證件相符。<br>根據香港法例第 486 章《個人資料（私隱）條例》，閣下有責任妥善處理及保管經手處理的個人資料。"
        
        if calculated_age is not None and calculated_age >= 18 and id_age_type == "兒童身份證":
            warning = "<span style='color: red; font-weight: bold; font-size: 14pt;'>持證人已年滿 18 歲並需要更換成人身份證，此證件已過期</span><br><br>"
            self.bottom_info_text.setHtml(warning + base_msg)
        else:
            self.bottom_info_text.setHtml(base_msg)



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
