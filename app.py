import threading
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ============================================================
# 穿刺角度ガイドシミュレータ Ver 4.1
#   - AI針検出: Teachable Machine で学習したモデル (ONNX) で
#     「穿刺針あり / なし」をリアルタイム判定
#   - 角度計測: OpenCV CSRT トラッカーによるロックオン追従 or
#     Hough変換による自動直線検出
#   - 採点機能: 記録した角度の平均・ばらつきからスコア算出
# ============================================================

st.set_page_config(page_title="穿刺角度ガイドシミュレータ", layout="wide")
st.title("💉 穿刺角度ガイドシミュレータ")
st.caption("Ver 4.0 — AI針検出 (Teachable Machine) × OpenCV追従 × 角度採点")

MODEL_PATH = Path(__file__).parent / "needle_model.onnx"
# Teachable Machine の学習クラス: index 0 = 穿刺(針あり), 1 = no 穿刺針
NEEDLE_CLASS_INDEX = 0
AI_THRESHOLD = 0.7  # この確率以上で「針あり」と判定


@st.cache_resource
def load_onnx_session():
    """ONNXモデルを読み込む。失敗してもアプリ自体は動かす。"""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name
        return sess, input_name
    except Exception as e:
        print(f"ONNX model load failed: {e}")
        return None, None


onnx_session, onnx_input_name = load_onnx_session()

# ------------------------------------------------------------
# サイドバー設定
# ------------------------------------------------------------
st.sidebar.header("📷 カメラ設定")
camera_mode = st.sidebar.radio(
    "カメラの向き", ("インカメラ (自分側)", "アウトカメラ (外側)"), index=1
)
facing = "user" if camera_mode == "インカメラ (自分側)" else "environment"
video_constraints = {
    "facingMode": facing,
    "width": {"ideal": 1280},
    "height": {"ideal": 720},
}

st.sidebar.markdown("---")
st.sidebar.header("🛠 検出モード")
mode = st.sidebar.radio(
    "モード選択", ("マーカー追従 (ロックオン方式)", "自動検出 (Hough変換)")
)

ai_enabled = st.sidebar.checkbox(
    "🤖 AI針検出を使う", value=onnx_session is not None,
    disabled=onnx_session is None,
    help="Teachable Machineで学習したモデルで針の有無を判定します",
)
if onnx_session is None:
    st.sidebar.warning("AIモデル (needle_model.onnx) が読み込めませんでした")

if mode == "自動検出 (Hough変換)":
    st.sidebar.subheader("自動検出設定")
    target_angle = st.sidebar.slider("目標角度 (°)", 10.0, 90.0, 30.0, step=1.0)
    roi_percent = st.sidebar.slider("ROIサイズ (%)", 30, 100, 60)
    hsv_threshold = st.sidebar.slider("HSV彩度閾値", 0, 255, 60)
else:
    st.sidebar.subheader("マーカー追従設定")
    st.sidebar.info(
        "手順:\n"
        "1. 映像の下の「📸 位置を選ぶ」を押す\n"
        "2. 画像を直接タップして【針先】→【根本】を指定\n"
        "   (🪄 自動検出も使えます)\n"
        "3. 「🎯 ロックオン」で追従開始"
    )
    if "tracking_active" not in st.session_state:
        st.session_state["tracking_active"] = False

    target_angle = st.sidebar.slider("目標角度 (°)", 10.0, 90.0, 30.0, step=1.0)
    roi_percent = 60
    hsv_threshold = 60

# 採点 (テスト) 機能
st.sidebar.markdown("---")
st.sidebar.header("📊 角度テスト (採点)")
if "is_recording" not in st.session_state:
    st.session_state["is_recording"] = False

rec_col1, rec_col2 = st.sidebar.columns(2)
with rec_col1:
    start_test = st.button("▶️ テスト開始")
with rec_col2:
    stop_test = st.button("⏹ 終了・採点")


# ------------------------------------------------------------
# 映像処理クラス
# ------------------------------------------------------------
class NeedleGuideProcessor(VideoProcessorBase):
    def __init__(self):
        # 共通
        self.mode = "Tracking"
        self.target_angle = 30.0
        self.angle_history = deque(maxlen=7)  # 角度の平滑化用 (中央値)
        self.lock = threading.Lock()

        # 採点用
        self.is_recording = False
        self.recorded_angles = []

        # AI判定
        self.ai_enabled = False
        self.session = onnx_session
        self.input_name = onnx_input_name
        self.frame_count = 0
        self.needle_prob = 0.0
        self.needle_detected = False

        # 自動検出パラメータ
        self.roi_percent = 60
        self.threshold = 60

        # 追従 (Tracking) パラメータ
        self.tracking_active = False
        self.tracker_tip = None
        self.tracker_tail = None
        self.track_init_done = False
        # 初期枠の相対座標 (x, y, w, h)
        self.box_tip_rel = (0.4, 0.4, 0.1, 0.1)
        self.box_tail_rel = (0.6, 0.6, 0.1, 0.1)
        # 位置選択用: 直近のクリーンなフレームと、選択時のスナップショット
        self.last_frame = None
        self.init_frame = None

    # ---------- UI からの設定反映 ----------
    def update_settings(self, mode, tgt_angle, roi_pct, thresh, ai_on,
                        tracking_active, recording):
        self.mode = mode
        self.target_angle = tgt_angle
        self.roi_percent = roi_pct
        self.threshold = thresh
        self.ai_enabled = ai_on and self.session is not None

        if mode == "Tracking":
            if tracking_active and not self.tracking_active:
                self.track_init_done = False  # 次フレームでトラッカー初期化
            self.tracking_active = tracking_active
        else:
            self.tracking_active = False

        if recording and not self.is_recording:
            with self.lock:
                self.recorded_angles = []
        self.is_recording = recording

    # ---------- AI針検出 (Teachable Machine 互換の前処理) ----------
    def classify(self, img_bgr):
        try:
            h, w = img_bgr.shape[:2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            crop = img_bgr[y0:y0 + side, x0:x0 + side]
            rgb = cv2.cvtColor(
                cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            x = (rgb.astype(np.float32) / 127.5) - 1.0
            out = self.session.run(None, {self.input_name: x[None]})[0][0]
            self.needle_prob = float(out[NEEDLE_CLASS_INDEX])
            self.needle_detected = self.needle_prob > AI_THRESHOLD
        except Exception as e:
            print(f"AI classify error: {e}")

    def draw_ai_status(self, img):
        h = img.shape[0]
        if self.needle_detected:
            text = f"AI Needle: {self.needle_prob * 100:.0f}%"
            color = (0, 200, 0)
        else:
            text = f"AI No Needle ({self.needle_prob * 100:.0f}%)"
            color = (0, 80, 255)
        cv2.putText(img, text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ---------- 角度の記録と平滑化 ----------
    def register_angle(self, angle):
        self.angle_history.append(angle)
        smoothed = float(np.median(self.angle_history))
        if self.is_recording:
            with self.lock:
                self.recorded_angles.append(smoothed)
        return smoothed

    def create_tracker(self):
        try:
            return cv2.TrackerCSRT_create()
        except AttributeError:
            pass
        try:
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            pass
        try:
            return cv2.TrackerMIL_create()
        except AttributeError:
            print("Error: No suitable tracker found.")
            return None

    # ---------- フレーム処理 ----------
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            height, width = img.shape[:2]

            # 位置選択用に描画前のクリーンなフレームを保持 (3フレームに1回)
            if self.frame_count % 3 == 0 or self.last_frame is None:
                self.last_frame = img.copy()

            # AI判定 (5フレームに1回、負荷軽減)
            self.frame_count += 1
            if self.ai_enabled and self.frame_count % 5 == 0:
                self.classify(img)

            if self.mode == "Tracking":
                self._process_tracking(img, width, height)
            else:
                self._process_hough(img, width, height)

            if self.ai_enabled:
                self.draw_ai_status(img)

            if self.is_recording:
                cv2.circle(img, (width - 25, 25), 10, (0, 0, 255), -1)
                cv2.putText(img, "REC", (width - 75, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"recv error: {e}")
            return frame

    # ---------- モードA: ロックオン追従 ----------
    def _process_tracking(self, img, width, height):
        tip_x = int(self.box_tip_rel[0] * width)
        tip_y = int(self.box_tip_rel[1] * height)
        tip_w = int(self.box_tip_rel[2] * width)
        tip_h = int(self.box_tip_rel[3] * height)
        tail_x = int(self.box_tail_rel[0] * width)
        tail_y = int(self.box_tail_rel[1] * height)
        tail_w = int(self.box_tail_rel[2] * width)
        tail_h = int(self.box_tail_rel[3] * height)

        if not self.tracking_active:
            # セットアップ画面: 下の画像タップUIで位置選択するよう案内
            cv2.putText(img, "Select needle points below, then Lock-on",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            self.angle_history.clear()
            return

        # ロックオン開始 (初回のみトラッカー初期化)
        # ユーザーが位置選択したスナップショットがあればそれで初期化し、
        # 選んだ場所に正確にロックオンする
        if not self.track_init_done:
            self.tracker_tip = self.create_tracker()
            self.tracker_tail = self.create_tracker()
            if self.tracker_tip and self.tracker_tail:
                init_img = self.init_frame if self.init_frame is not None else img
                self.tracker_tip.init(init_img, (tip_x, tip_y, tip_w, tip_h))
                self.tracker_tail.init(init_img, (tail_x, tail_y, tail_w, tail_h))
                self.track_init_done = True

        if not self.track_init_done:
            return

        ok_tip, bbox_tip = self.tracker_tip.update(img)
        ok_tail, bbox_tail = self.tracker_tail.update(img)

        if ok_tip and ok_tail:
            p1 = (int(bbox_tip[0] + bbox_tip[2] / 2),
                  int(bbox_tip[1] + bbox_tip[3] / 2))
            p2 = (int(bbox_tail[0] + bbox_tail[2] / 2),
                  int(bbox_tail[1] + bbox_tail[3] / 2))

            cv2.rectangle(img, (int(bbox_tip[0]), int(bbox_tip[1])),
                          (int(bbox_tip[0] + bbox_tip[2]),
                           int(bbox_tip[1] + bbox_tip[3])), (255, 0, 0), 2)
            cv2.rectangle(img, (int(bbox_tail[0]), int(bbox_tail[1])),
                          (int(bbox_tail[0] + bbox_tail[2]),
                           int(bbox_tail[1] + bbox_tail[3])), (0, 0, 255), 2)
            cv2.line(img, p1, p2, (0, 255, 255), 4)

            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            angle = 90.0 if dx == 0 else float(
                np.degrees(np.arctan2(abs(dy), abs(dx))))
            smoothed = self.register_angle(angle)

            on_target = abs(smoothed - self.target_angle) < 5.0
            color = (0, 255, 0) if on_target else (0, 255, 255)
            cv2.putText(img, f"Angle: {smoothed:.1f} / {self.target_angle:.0f}",
                        (p1[0], p1[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(img, "Tracking Lost", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---------- モードB: Hough変換による自動検出 ----------
    def _process_hough(self, img, width, height):
        roi_w = int(width * (self.roi_percent / 100))
        roi_h = int(height * (self.roi_percent / 100))
        roi_x = (width - roi_w) // 2
        roi_y = (height - roi_h) // 2
        roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        _, saturation_mask = cv2.threshold(
            s, self.threshold, 255, cv2.THRESH_BINARY_INV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_v = clahe.apply(v)
        processed = cv2.bitwise_and(enhanced_v, enhanced_v, mask=saturation_mask)
        blurred = cv2.GaussianBlur(processed, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=40, maxLineGap=30)

        best_line, max_len, current_angle = None, 0.0, 0.0
        if lines is not None:
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                la = 90.0 if lx2 == lx1 else float(
                    np.degrees(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1))))
                if 5 < la < 85:
                    length = float(np.hypot(lx2 - lx1, ly2 - ly1))
                    if length > max_len:
                        max_len = length
                        best_line = (lx1 + roi_x, ly1 + roi_y,
                                     lx2 + roi_x, ly2 + roi_y)
                        current_angle = la

        if best_line is not None:
            bx1, by1, bx2, by2 = best_line
            tip, tail = (((bx1, by1), (bx2, by2)) if by1 > by2
                         else ((bx2, by2), (bx1, by1)))
            smoothed = self.register_angle(current_angle)
            on_target = abs(smoothed - self.target_angle) < 5.0
            color = (0, 255, 0) if on_target else (0, 255, 255)
            cv2.line(img, tail, tip, color, 4)
            cv2.putText(img, f"{smoothed:.1f}", (tip[0], tip[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.rectangle(img, (roi_x, roi_y),
                      (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)


# ------------------------------------------------------------
# WebRTC 接続設定
#   ローカルや同一ネットワークでは STUN だけで直接接続できる。
#   Streamlit Cloud などで NAT 越えが必要な端末向けに、TURN は
#   st.secrets に設定があるときだけ追加する (動作する TURN を各自用意)。
#   ※ かつての無料 Open Relay TURN は廃止されており、設定すると
#     応答待ちで接続がタイムアウトするため使用しない。
# ------------------------------------------------------------
ice_servers = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
]
try:
    turn = st.secrets["turn"]
    ice_servers.append({
        "urls": turn["urls"],
        "username": turn["username"],
        "credential": turn["credential"],
    })
except Exception:
    pass

RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})

ctx = webrtc_streamer(
    key="needle-guide-v4",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=NeedleGuideProcessor,
    media_stream_constraints={"video": video_constraints, "audio": False},
    async_processing=True,
)


# ------------------------------------------------------------
# マーカー追従: 画像タップによる針の位置選択
# ------------------------------------------------------------
def detect_needle_line(img_bgr):
    """Hough変換でスナップショット上の針らしき直線を探し (tip, tail) を返す"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=80, maxLineGap=20)
    best, max_len = None, 0.0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ang = 90.0 if x2 == x1 else float(
                np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
            if 5 < ang < 85:
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length > max_len:
                    max_len, best = length, (x1, y1, x2, y2)
    if best is None:
        return None
    x1, y1, x2, y2 = best
    # 画面下側の端点を針先とみなす
    if y1 > y2:
        return (int(x1), int(y1)), (int(x2), int(y2))
    return (int(x2), int(y2)), (int(x1), int(y1))


def _clear_selection():
    for k in ("snapshot", "sel_tip", "sel_tail", "picker_last_click"):
        st.session_state.pop(k, None)


if mode == "マーカー追従 (ロックオン方式)":
    st.markdown("#### 🎯 針の位置選択")

    if st.session_state["tracking_active"]:
        st.success("🔒 追従中です。位置を選び直すには「リセット」を押してください。")
        if st.button("🔄 リセット (選び直す)", use_container_width=True):
            st.session_state["tracking_active"] = False
            _clear_selection()
            st.rerun()
    else:
        playing = bool(ctx.state.playing) if ctx.state else False
        cap_col, auto_col = st.columns(2)
        with cap_col:
            if st.button("📸 位置を選ぶ (今の映像を切り取る)",
                         disabled=not playing, use_container_width=True):
                frame = (ctx.video_processor.last_frame
                         if ctx.video_processor else None)
                if frame is None:
                    st.warning("映像がまだ届いていません。少し待ってからお試しください。")
                else:
                    _clear_selection()
                    st.session_state["snapshot"] = frame.copy()

        snap = st.session_state.get("snapshot")

        with auto_col:
            if st.button("🪄 自動で針に合わせる",
                         disabled=snap is None, use_container_width=True):
                result = detect_needle_line(snap)
                if result:
                    st.session_state["sel_tip"], st.session_state["sel_tail"] = result
                    st.session_state["picker_last_click"] = None
                else:
                    st.warning("針らしい直線が見つかりませんでした。画像をタップして手動で指定してください。")

        if not playing and snap is None:
            st.info("カメラをSTARTしてから「📸 位置を選ぶ」を押してください")

        if snap is not None:
            tip_pt = st.session_state.get("sel_tip")
            tail_pt = st.session_state.get("sel_tail")

            if tip_pt is None:
                st.info("👆 画像上で【針先】をタップしてください")
            elif tail_pt is None:
                st.info("👆 次に【針の根本】をタップしてください")
            else:
                st.success("✅ 位置OK!「ロックオン」を押すと追従開始 (もう一度タップすると針先からやり直し)")

            # 選択状況を描画して表示
            disp = snap.copy()
            if tip_pt is not None:
                cv2.circle(disp, tip_pt, 16, (255, 80, 0), 3)
                cv2.putText(disp, "TIP", (tip_pt[0] + 20, tip_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 0), 2)
            if tail_pt is not None:
                cv2.circle(disp, tail_pt, 16, (0, 0, 255), 3)
                cv2.putText(disp, "BODY", (tail_pt[0] + 20, tail_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if tip_pt is not None and tail_pt is not None:
                cv2.line(disp, tip_pt, tail_pt, (0, 255, 255), 3)

            value = streamlit_image_coordinates(
                Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)),
                key="needle_picker",
                use_column_width="always",
            )
            if value is not None:
                click = (int(value["x"]), int(value["y"]))
                if click != st.session_state.get("picker_last_click"):
                    st.session_state["picker_last_click"] = click
                    if tip_pt is None:
                        st.session_state["sel_tip"] = click
                    elif tail_pt is None:
                        st.session_state["sel_tail"] = click
                    else:
                        # 3回目のタップは針先の選び直し
                        st.session_state["sel_tip"] = click
                        st.session_state["sel_tail"] = None
                    st.rerun()

            both_set = (st.session_state.get("sel_tip") is not None
                        and st.session_state.get("sel_tail") is not None)
            if st.button("🎯 この位置でロックオン", type="primary",
                         disabled=not both_set, use_container_width=True):
                if ctx.video_processor:
                    h0, w0 = snap.shape[:2]
                    tp = st.session_state["sel_tip"]
                    tl = st.session_state["sel_tail"]
                    # 枠サイズは針の長さに応じて自動調整
                    dist = float(np.hypot(tp[0] - tl[0], tp[1] - tl[1]))
                    side = int(min(max(0.5 * dist, 24), 96))

                    def to_rel_box(pt):
                        x = min(max(pt[0] - side / 2, 0), w0 - side)
                        y = min(max(pt[1] - side / 2, 0), h0 - side)
                        return (x / w0, y / h0, side / w0, side / h0)

                    proc = ctx.video_processor
                    proc.box_tip_rel = to_rel_box(tp)
                    proc.box_tail_rel = to_rel_box(tl)
                    proc.init_frame = snap  # 選択した画像でトラッカー初期化
                    st.session_state["tracking_active"] = True
                    st.rerun()
                else:
                    st.warning("カメラが動作していません。STARTしてからお試しください。")


# ------------------------------------------------------------
# テスト (採点) 制御と結果表示
# ------------------------------------------------------------
if start_test:
    st.session_state["is_recording"] = True
    st.session_state.pop("test_result", None)
if stop_test and st.session_state["is_recording"]:
    st.session_state["is_recording"] = False
    if ctx.video_processor:
        with ctx.video_processor.lock:
            angles = list(ctx.video_processor.recorded_angles)
        st.session_state["test_result"] = angles

# パラメータ動的反映
if ctx.video_processor:
    proc_mode = "Tracking" if mode == "マーカー追従 (ロックオン方式)" else "Auto"
    ctx.video_processor.update_settings(
        proc_mode,
        target_angle,
        roi_percent,
        hsv_threshold,
        ai_enabled,
        st.session_state.get("tracking_active", False),
        st.session_state["is_recording"],
    )

if st.session_state["is_recording"]:
    st.info("📹 記録中... 穿刺動作を行い、終わったら「終了・採点」を押してください")

if "test_result" in st.session_state:
    angles = st.session_state["test_result"]
    st.markdown("---")
    st.subheader("🏆 テスト結果")
    if len(angles) < 10:
        st.warning("データ不足です (10サンプル以上必要)。もう一度お試しください。")
    else:
        avg = float(np.mean(angles))
        std = float(np.std(angles))
        diff = abs(avg - target_angle)
        score = max(0, round(100 - diff * 2 - std * 5))
        message = ("素晴らしい！" if score >= 90
                   else "良好です" if score >= 70 else "練習が必要です")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("スコア", f"{score} 点")
        m2.metric("平均角度", f"{avg:.1f}°", f"目標 {target_angle:.0f}°")
        m3.metric("安定性 (±SD)", f"{std:.2f}°")
        m4.metric("サンプル数", len(angles))
        st.success(message)
        st.line_chart(angles, height=200)
