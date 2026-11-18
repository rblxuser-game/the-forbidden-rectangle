import os, sys, time, ctypes, threading
from operator import itemgetter
import cv2, numpy as np, torch, mss
from ultralytics import YOLO
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QIcon, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QSystemTrayIcon, QMenu,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
)

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

try:
    import bettercam
    HAS_DX = True
except ImportError:
    HAS_DX = False


class RTXOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        geo = QApplication.primaryScreen().geometry()
        self.setGeometry(geo)
        self.w_px, self.h_px = geo.width(), geo.height()

        u32 = ctypes.windll.user32
        mw, mh = u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)
        self.ox, self.oy = 0, 0
        self.ltrb = (0, 0, mw, mh)
        self.gmss = {"left": 0, "top": 0, "width": mw, "height": mh}
        self.mx, self.my = self.w_px / max(mw, 1), self.h_px / max(mh, 1)

        self.gpu = torch.cuda.is_available()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.best_path = os.path.join(self.base_dir, "best.pt")
        self.yolo_path = os.path.join(self.base_dir, "yolov8n.pt")

        self.blocky_mode = os.path.exists(self.best_path)
        self.model = None
        self.loaded_mode = None

        self.hw = f"{'RTX 3050' if self.gpu else 'CPU'} [LOADING...]"
        self.lock = threading.Lock()
        self.objs, self.render_objs, self.nid = {}, {}, 1

        self.hold_sec = 0.50
        self.mdist = 850.0
        self.sm_pos = 0.82
        self.sm_w = 0.22
        self.sm_h = 0.20
        self.deadband_px = 1.5
        self.last_paint = time.perf_counter()

        self.flash_enabled = True
        self.screen_white = False
        self.flash_alpha = 0.0
        self.flash_pixmap = None
        self._auto_load_flash_image()

        self.paused, self.run, self.rec = False, True, False
        self.show_lines = True
        self.saved, self.fps, self.ms = 0, 0.0, 0.0

        self._init_tray()
        threading.Thread(target=self._loop, daemon=True).start()

        self.tmr = QTimer(self)
        self.tmr.timeout.connect(self.update)
        self.tmr.start(8)

    def _auto_load_flash_image(self):
        for name in ("GGS.png", "GGS.jpg", "flash.png", "flash.jpg", "flash.jpeg", "flash.webp"):
            p = os.path.join(self.base_dir, name)
            if os.path.exists(p):
                self.set_flash_image(p)
                break

    def set_flash_image(self, path):
        pix = QPixmap(path)
        if not pix.isNull():
            with self.lock:
                self.flash_pixmap = pix
            self.update()
            return True
        return False

    def _is_screen_white(self, frame):
        small = cv2.resize(frame, (32, 18))
        return float(np.mean(small)) > 220.0 and float(np.mean(small > 235)) > 0.75

    def _load_active_model(self):
        want_blocky = self.blocky_mode and os.path.exists(self.best_path)
        w_path = self.best_path if want_blocky else self.yolo_path
        m = YOLO(w_path)
        if self.gpu:
            m.to("cuda:0")
            try:
                m.model.half()
            except Exception:
                pass
        self.model = m
        self.loaded_mode = self.blocky_mode

    def _is_ui_or_inspect_screen(self, frame):
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (64, 36))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        if np.mean((sat < 35) & (val > 25) & (val < 85)) > 0.58:
            return True
        top_strip = frame[0:max(1, int(fh * 0.08)), int(fw * 0.25):int(fw * 0.75)]
        return top_strip.size > 0 and np.mean(top_strip > 235) > 0.14

    def _is_valid_target(self, x, y, w, h):
        if w < 16 or h < 22 or w > self.w_px * 0.45 or h > self.h_px * 0.78:
            return False

        aspect = w / max(h, 1.0)
        if self.blocky_mode:
            if aspect < 0.28 or aspect > 1.35:
                return False
        else:
            if aspect < 0.20 or aspect > 0.95:
                return False

        cx = (x + w * 0.5) / max(self.w_px, 1)
        cy = (y + h * 0.5) / max(self.h_px, 1)

        # Top-left HUD deadzone (Roblox buttons, radar minimap, & top UI text)
        if cx < 0.24 and cy < 0.34:
            return False
        if cx < 0.42 and cy < 0.08:
            return False

        if (y + h) > self.h_px * 0.95 and h < self.h_px * 0.35:
            return False

        return not ((h / max(self.h_px, 1)) > 0.34 and (0.24 <= cx <= 0.47) and (cy >= 0.48))

    def _init_tray(self):
        pix = QPixmap(16, 16)
        pix.fill(QColor(255, 45, 45))
        self.tray = QSystemTrayIcon(QIcon(pix), self)
        menu = QMenu()
        for label, fn in [
            ("Toggle Overlay", self.toggle_pause),
            ("Toggle Blocky / Humanoid", self.toggle_model_mode),
            ("Toggle Flash Image", self.toggle_flash),
            ("Toggle Dataset Recording", self.toggle_rec),
        ]:
            act = QAction(label, self)
            act.triggered.connect(fn)
            menu.addAction(act)
        menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self._quit)
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _quit(self):
        self.run = False
        self.tray.hide()
        QApplication.quit()

    def showEvent(self, e):
        super().showEvent(e)
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x11)
        except Exception:
            pass

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            if self.paused:
                self.objs.clear()
                self.render_objs.clear()
                self.screen_white = False
                self.flash_alpha = 0.0
        self.update()
        return not self.paused

    def toggle_model_mode(self):
        with self.lock:
            self.blocky_mode = not self.blocky_mode
            self.objs.clear()
            self.render_objs.clear()
        self.update()
        return self.blocky_mode

    def toggle_flash(self):
        with self.lock:
            self.flash_enabled = not self.flash_enabled
            if not self.flash_enabled:
                self.screen_white = False
                self.flash_alpha = 0.0
        self.update()
        return self.flash_enabled

    def toggle_lines(self):
        with self.lock:
            self.show_lines = not self.show_lines
        self.update()
        return self.show_lines

    def toggle_rec(self):
        with self.lock:
            self.rec = not self.rec
            if self.rec:
                os.makedirs(os.path.join(self.base_dir, "dataset", "images"), exist_ok=True)
                os.makedirs(os.path.join(self.base_dir, "dataset", "labels"), exist_ok=True)
        self.update()
        return self.rec
    def _loop(self):
        lt = time.perf_counter()
        cam, mode = None, "MSS"
        if HAS_DX:
            for dev_idx in (0, 1):
                try:
                    cam = bettercam.create(device_idx=dev_idx, output_color="BGR")
                    reg = (0, 0, cam.width, cam.height)
                    if cam.grab(region=reg) is not None:
                        self.ox, self.oy = 0, 0
                        self.ltrb = reg
                        self.mx = self.w_px / max(cam.width, 1)
                        self.my = self.h_px / max(cam.height, 1)
                        mode = f"DX{dev_idx}"
                        break
                    cam.start(region=reg, target_fps=144)
                    self.ox, self.oy = 0, 0
                    self.ltrb = reg
                    self.mx = self.w_px / max(cam.width, 1)
                    self.my = self.h_px / max(cam.height, 1)
                    mode = f"DX{dev_idx}-A"
                    break
                except Exception as ex:
                    print(f"BetterCam dev {dev_idx} fallback:", ex)
                    cam = None

        sct = None if mode.startswith("DX") else mss.MSS()
        isz = 480 if self.gpu else 256
        step = 0
        img_dir = os.path.join(self.base_dir, "dataset", "images")
        lbl_dir = os.path.join(self.base_dir, "dataset", "labels")

        with torch.inference_mode():
            while self.run:
                if self.model is None or self.loaded_mode != self.blocky_mode:
                    self.hw = f"{'RTX 3050' if self.gpu else 'CPU'} [{mode}|SWITCHING]"
                    self._load_active_model()
                    tag = "BLOCKY" if (self.blocky_mode and os.path.exists(self.best_path)) else "HUMANOID"
                    self.hw = f"{'RTX 3050' if self.gpu else 'CPU'} [{mode}|{tag}]"

                if self.paused:
                    time.sleep(0.05)
                    continue

                t0 = time.perf_counter()
                if mode.startswith("DX") and not mode.endswith("-A"):
                    frame = cam.grab(region=self.ltrb)
                    if frame is None:
                        continue
                elif mode.endswith("-A"):
                    frame = cam.get_latest_frame()
                    if frame is None:
                        time.sleep(0.001)
                        continue
                else:
                    frame = np.ascontiguousarray(np.asarray(sct.grab(self.gmss))[:, :, :3])

                if self.flash_enabled and self._is_screen_white(frame):
                    with self.lock:
                        self.screen_white = True
                        self.objs.clear()
                    time.sleep(0.015)
                    continue
                elif self.screen_white:
                    with self.lock:
                        self.screen_white = False

                if self._is_ui_or_inspect_screen(frame):
                    with self.lock:
                        self.objs.clear()
                    time.sleep(0.04)
                    continue

                cf = 0.24 if (self.blocky_mode and os.path.exists(self.best_path)) else 0.16
                res = self.model.predict(
                    frame, classes=0, conf=cf, iou=0.50, imgsz=isz,
                    device=0 if self.gpu else "cpu", verbose=False
                )

                dets, ylines = [], []
                fh, fw = frame.shape[:2]
                max_w_ratio = 1.25 if self.blocky_mode else 0.90

                for r in res:
                    for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
                        bw, bh = x2 - x1, y2 - y1
                        w, h = bw * self.mx, bh * self.my
                        x, y = (x1 + self.ox) * self.mx, (y1 + self.oy) * self.my

                        if not self._is_valid_target(x, y, w, h):
                            continue

                        max_w = h * max_w_ratio
                        if w > max_w:
                            x += (w - max_w) * 0.5
                            w = max_w

                        dets.append((x, y, w, h, x + w * 0.5, y + h * 0.5))
                        if self.rec:
                            ylines.append(
                                f"0 {((x1 + x2) / 2) / fw:.6f} {((y1 + y2) / 2) / fh:.6f} {bw / fw:.6f} {bh / fh:.6f}"
                            )

                step += 1
                if self.rec and ylines and step % 4 == 0:
                    fn = f"cap_{int(time.time() * 1000)}"
                    cv2.imwrite(os.path.join(img_dir, f"{fn}.jpg"), frame)
                    with open(os.path.join(lbl_dir, f"{fn}.txt"), "w") as f:
                        f.write("\n".join(ylines))
                    with self.lock:
                        self.saved += 1

                now = time.perf_counter()
                ims = (now - t0) * 1000.0
                ifps = 1.0 / max(now - lt, 1e-5)
                lt = now

                with self.lock:
                    self._match(dets)
                    self.fps = self.fps * 0.85 + ifps * 0.15
                    self.ms = self.ms * 0.85 + ims * 0.15
    def _match(self, dets):
        now = time.perf_counter()
        if not dets:
            for k in list(self.objs):
                self.objs[k]["g"] += 1
                self.objs[k]["vx"] *= 0.35
                self.objs[k]["vy"] *= 0.35
                if (now - self.objs[k]["seen"]) > self.hold_sec:
                    del self.objs[k]
            return

        pairs = []
        max_match_dist = self.mdist
        for i, (nx, ny, nw, nh, cx, cy) in enumerate(dets):
            for k, o in self.objs.items():
                dt = min(now - o["t"], 0.08)
                ocx, ocy = o["c"]
                px = ocx + o["vx"] * dt
                py = ocy + o["vy"] * dt

                dx = (cx - px) * 0.45
                raw_dy = cy - py
                dy = raw_dy * (0.55 if raw_dy > 0 else 0.95)
                d = (dx * dx + dy * dy) ** 0.5

                ox, oy, ow, oh = o["b"]
                ix1, iy1 = max(nx, ox), max(ny, oy)
                ix2, iy2 = min(nx + nw, ox + ow), min(ny + nh, oy + oh)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter > 0.0:
                    union = nw * nh + ow * oh - inter
                    iou = inter / max(union, 1.0)
                    d *= (1.0 - 0.55 * iou)

                if d <= max_match_dist:
                    pairs.append((d, i, k))

        pairs.sort(key=itemgetter(0))
        ud, uk = set(), set()

        for _, i, k in pairs:
            if i in ud or k in uk:
                continue
            ud.add(i)
            uk.add(k)

            nx, ny, nw, nh, ncx, ncy = dets[i]
            ox, oy, ow, oh = self.objs[k]["b"]
            ocx, ocy = self.objs[k]["c"]
            dt = max(now - self.objs[k]["t"], 1e-4)

            dist = ((ncx - ocx) ** 2 + (ncy - ocy) ** 2) ** 0.5
            if dist < self.deadband_px:
                pa = 0.12
            elif dist > 20.0:
                pa = 0.94
            else:
                pa = min(0.92, max(self.sm_pos, dist / 50.0))

            short_streak = self.objs[k].get("short", 0)
            if nh < oh * 0.86:
                short_streak += 1
                ha = 0.50 if short_streak >= 3 else 0.12
            else:
                short_streak = 0
                ha = 0.38 if nh > oh * 1.15 else self.sm_h

            wa = min(0.65, max(self.sm_w, dist / 180.0))

            scx = ocx * (1.0 - pa) + ncx * pa
            top_y = oy * (1.0 - pa) + ny * pa
            sw = ow * (1.0 - wa) + nw * wa
            sh = oh * (1.0 - ha) + nh * ha

            sx = scx - sw * 0.5
            sy = top_y
            scy = sy + sh * 0.5

            if self.objs[k]["g"] > 0 or dist < self.deadband_px:
                vx, vy = 0.0, 0.0
            else:
                raw_vx = (scx - ocx) / dt
                raw_vy = (sy - oy) / dt
                vx = self.objs[k]["vx"] * 0.50 + raw_vx * 0.50
                vy = self.objs[k]["vy"] * 0.50 + raw_vy * 0.50

            self.objs[k] = {
                "b": (sx, sy, sw, sh),
                "c": (scx, scy),
                "vx": vx,
                "vy": vy,
                "t": now,
                "seen": now,
                "g": 0,
                "short": short_streak,
            }

        for k in list(self.objs):
            if k not in uk:
                self.objs[k]["g"] += 1
                dt = min(now - self.objs[k]["t"], 0.035)
                ox, oy, ow, oh = self.objs[k]["b"]
                ocx, ocy = self.objs[k]["c"]
                vx, vy = self.objs[k]["vx"] * 0.35, self.objs[k]["vy"] * 0.35
                self.objs[k]["b"] = (ox + vx * dt, oy + vy * dt, ow, oh)
                self.objs[k]["c"] = (ocx + vx * dt, ocy + vy * dt)
                self.objs[k]["vx"], self.objs[k]["vy"], self.objs[k]["t"] = vx, vy, now
                if (now - self.objs[k]["seen"]) > self.hold_sec:
                    del self.objs[k]

        for i, (x, y, w, h, cx, cy) in enumerate(dets):
            if i not in ud:
                k = self.nid
                self.nid += 1
                self.objs[k] = {
                    "b": (float(x), float(y), float(w), float(h)),
                    "c": (float(cx), float(cy)),
                    "vx": 0.0,
                    "vy": 0.0,
                    "t": now,
                    "seen": now,
                    "g": 0,
                    "short": 0,
                }

    @staticmethod
    def _spring_step(val, vel, target, omega, dt):
        f = 1.0 + 2.0 * dt * omega
        oo = omega * omega
        hoo = dt * oo
        hhoo = dt * hoo
        det_inv = 1.0 / (f + hhoo)
        new_val = (f * val + dt * vel + hhoo * target) * det_inv
        new_vel = (vel + hoo * (target - val)) * det_inv
        return new_val, new_vel

    def paintEvent(self, event):
        now = time.perf_counter()
        dt = min(now - self.last_paint, 0.05)
        self.last_paint = now

        with self.lock:
            target_objs = {k: v.copy() for k, v in self.objs.items()}
            fps, ms, hw = self.fps, self.ms, self.hw
            paused, rec, saved = self.paused, self.rec, self.saved
            show_lines = self.show_lines
            flash_on = self.flash_enabled
            is_white = self.screen_white and flash_on
            flash_pix = self.flash_pixmap

        if paused:
            self.render_objs.clear()
            return

        if not flash_on:
            self.flash_alpha = 0.0
        elif is_white:
            self.flash_alpha = 1.0
            self.render_objs.clear()
        elif self.flash_alpha > 0.0:
            self.flash_alpha = max(0.0, self.flash_alpha - dt * 1.65)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if not is_white:
            for k in list(self.render_objs):
                if k not in target_objs:
                    del self.render_objs[k]

            # Critically damped spring without overshoot lead (eliminates micro-stutter)
            omega_pos, omega_size = 78.0, 32.0
            for k, obj in target_objs.items():
                tx, ty, tw, th = obj["b"]
                if k not in self.render_objs:
                    self.render_objs[k] = [tx, ty, tw, th, 0.0, 0.0, 0.0, 0.0]
                else:
                    rx, ry, rw, rh, vx, vy, vw, vh = self.render_objs[k]
                    rx, vx = self._spring_step(rx, vx, tx, omega_pos, dt)
                    ry, vy = self._spring_step(ry, vy, ty, omega_pos, dt)
                    rw, vw = self._spring_step(rw, vw, tw, omega_size, dt)
                    rh, vh = self._spring_step(rh, vh, th, omega_size, dt)
                    self.render_objs[k] = [rx, ry, rw, rh, vx, vy, vw, vh]

            sc_x, sc_y = self.width() * 0.5, self.height() * 0.5
            if show_lines:
                cross_size = 14.0
                p.setPen(QPen(QColor(255, 45, 45, 51), 1.5))
                p.drawLine(QPointF(sc_x - cross_size, sc_y), QPointF(sc_x + cross_size, sc_y))
                p.drawLine(QPointF(sc_x, sc_y - cross_size), QPointF(sc_x, sc_y + cross_size))

            box_pen = QPen(QColor(255, 45, 45, 220), 2.0)
            line_pen = QPen(QColor(255, 45, 45, 110), 1.2)

            for k, (rx, ry, rw, rh, _, _, _, _) in self.render_objs.items():
                cx, cy = rx + rw * 0.5, ry + rh * 0.5
                if show_lines:
                    p.setPen(line_pen)
                    p.drawLine(QPointF(sc_x, sc_y), QPointF(cx, cy))
                p.setPen(box_pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(rx, ry, rw, rh))

            p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            status = f"{hw} | {fps:.0f} FPS ({ms:.1f}ms) | Objs: {len(self.render_objs)}"
            if rec:
                status += f" | REC [{saved}]"

            p.setPen(QColor(0, 0, 0, 180))
            p.drawText(17, 25, status)
            p.setPen(QColor(255, 60, 60))
            p.drawText(16, 24, status)

        if flash_on and self.flash_alpha > 0.0:
            p.save()
            p.setOpacity(self.flash_alpha * self.flash_alpha)
            if flash_pix is not None and not flash_pix.isNull():
                p.drawPixmap(self.rect(), flash_pix)
            else:
                p.fillRect(self.rect(), QColor(20, 22, 26, 235))
                p.setPen(QColor(255, 60, 60))
                p.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "FLASH DETECTED (GGS.png not found)")
            p.restore()
class ControlPanel(QWidget):
    def __init__(self, overlay: RTXOverlay):
        super().__init__()
        self.overlay = overlay
        self.setWindowTitle("RTX Overlay Control")
        self.setFixedSize(310, 340)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self.setStyleSheet("""
            QWidget {
                background-color: #14161a;
                color: #e6e8ec;
                font-family: 'Segoe UI', Consolas, sans-serif;
            }
            QLabel#Title {
                font-size: 14px;
                font-weight: bold;
                color: #ff3b3b;
            }
            QLabel#Stats {
                background-color: #1c1f26;
                border: 1px solid #2a2e38;
                border-radius: 6px;
                padding: 8px;
                font-family: Consolas;
                font-size: 11px;
                color: #9aa4b2;
            }
            QPushButton {
                background-color: #222630;
                border: 1px solid #323846;
                border-radius: 6px;
                padding: 9px;
                font-size: 12px;
                font-weight: bold;
                color: #d8dee9;
            }
            QPushButton:hover {
                background-color: #2b303d;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("OVERLAY CONTROLLER")
        title.setObjectName("Title")
        layout.addWidget(title)

        self.stats_lbl = QLabel("Starting UI...")
        self.stats_lbl.setObjectName("Stats")
        layout.addWidget(self.stats_lbl)

        self.btn_toggle = QPushButton("OVERLAY: ON")
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self.btn_toggle)

        self.btn_mode = QPushButton("MODE: BLOCKY (best.pt)")
        self.btn_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode.clicked.connect(self._on_mode)
        layout.addWidget(self.btn_mode)

        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self.btn_lines = QPushButton("LINES: ON")
        self.btn_lines.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lines.clicked.connect(self._on_lines)
        row1.addWidget(self.btn_lines)

        self.btn_rec = QPushButton("REC: OFF")
        self.btn_rec.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rec.clicked.connect(self._on_rec)
        row1.addWidget(self.btn_rec)

        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self.btn_flash_toggle = QPushButton("FLASH: ON")
        self.btn_flash_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_flash_toggle.clicked.connect(self._on_flash_toggle)
        row2.addWidget(self.btn_flash_toggle)

        self.btn_flash_img = QPushButton("IMG: GGS.png")
        self.btn_flash_img.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_flash_img.clicked.connect(self._on_pick_flash)
        row2.addWidget(self.btn_flash_img)

        layout.addLayout(row2)

        self.btn_quit = QPushButton("EXIT")
        self.btn_quit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_quit.setStyleSheet("background-color: #2a181a; border-color: #5c262b; color: #ff6b6b;")
        self.btn_quit.clicked.connect(self._on_quit)
        layout.addWidget(self.btn_quit)

        self._refresh_styles()

        self.stat_tmr = QTimer(self)
        self.stat_tmr.timeout.connect(self._update_stats)
        self.stat_tmr.start(200)

    def _refresh_styles(self):
        if not self.overlay.paused:
            self.btn_toggle.setText("OVERLAY: ON")
            self.btn_toggle.setStyleSheet(
                "background-color: #3a1619; border: 1px solid #ff3b3b; color: #ff5c5c; font-size: 13px;"
            )
        else:
            self.btn_toggle.setText("OVERLAY: OFF")
            self.btn_toggle.setStyleSheet(
                "background-color: #1c1f26; border: 1px solid #323846; color: #7b8494; font-size: 13px;"
            )

        if self.overlay.blocky_mode:
            self.btn_mode.setText("MODE: BLOCKY (best.pt)")
            self.btn_mode.setStyleSheet(
                "background-color: #192938; border: 1px solid #3b82f6; color: #60a5fa;"
            )
        else:
            self.btn_mode.setText("MODE: HUMANOID (yolov8n)")
            self.btn_mode.setStyleSheet(
                "background-color: #1a2e26; border: 1px solid #10b981; color: #34d399;"
            )

        if self.overlay.show_lines:
            self.btn_lines.setText("LINES: ON")
            self.btn_lines.setStyleSheet("background-color: #281a1d; border-color: #a8323b; color: #ff7575;")
        else:
            self.btn_lines.setText("LINES: OFF")
            self.btn_lines.setStyleSheet("")

        if self.overlay.rec:
            self.btn_rec.setText("REC: ON")
            self.btn_rec.setStyleSheet("background-color: #3b2414; border-color: #d97724; color: #ffaa5c;")
        else:
            self.btn_rec.setText("REC: OFF")
            self.btn_rec.setStyleSheet("")

        if self.overlay.flash_enabled:
            self.btn_flash_toggle.setText("FLASH: ON")
            self.btn_flash_toggle.setStyleSheet("background-color: #262238; border-color: #6d5cae; color: #b4a7f5;")
        else:
            self.btn_flash_toggle.setText("FLASH: OFF")
            self.btn_flash_toggle.setStyleSheet("")

        if self.overlay.flash_pixmap is not None:
            self.btn_flash_img.setText("IMG: LOADED")
            self.btn_flash_img.setStyleSheet("background-color: #222834; border-color: #4b5563; color: #cbd5e1;")

    def _on_toggle(self):
        self.overlay.toggle_pause()
        self._refresh_styles()

    def _on_mode(self):
        self.overlay.toggle_model_mode()
        self._refresh_styles()

    def _on_lines(self):
        self.overlay.toggle_lines()
        self._refresh_styles()

    def _on_rec(self):
        self.overlay.toggle_rec()
        self._refresh_styles()

    def _on_flash_toggle(self):
        self.overlay.toggle_flash()
        self._refresh_styles()

    def _on_pick_flash(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select White-Screen Image", self.overlay.base_dir,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            self.overlay.set_flash_image(path)
            self._refresh_styles()

    def _on_quit(self):
        self.overlay._quit()

    def _update_stats(self):
        self._refresh_styles()
        with self.overlay.lock:
            hw = self.overlay.hw
            fps = self.overlay.fps
            ms = self.overlay.ms
            objs = len(self.overlay.render_objs)
            paused = self.overlay.paused
            saved = self.overlay.saved

        if paused:
            self.stats_lbl.setText(f"{hw}\nStatus: STANDBY (Paused)")
        else:
            self.stats_lbl.setText(f"{hw}\n{fps:.0f} FPS ({ms:.1f} ms) | Targets: {objs} | Saved: {saved}")

    def closeEvent(self, event):
        self.overlay._quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = RTXOverlay()
    overlay.show()
    panel = ControlPanel(overlay)
    panel.show()
    sys.exit(app.exec())
