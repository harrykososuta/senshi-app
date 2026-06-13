import threading
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ============================================================
# 穿刺角度ガイドシミュレータ Ver 4.0
#   - AI針検出: Teachable Machine で学習したモデル (ONNX) で
#     「穿刺針あり / なし」をリアルタイム判定
#   - 角度計測: OpenCV CSRT トラッカーによるロックオン追従 or
#     Hough変換による自動直線検出
#   - 採点機能: 記録した角度の平均・ばらつきからスコア算出
# ============================================================

st.set_page_config(page_title="穿刺角度ガイドシミュレータ", layout="wide")
st.title("💉 穿刺角度ガイドシミュレータ")
st.caption("Ver 4.0 — AI針検出 (Teachable Machine) × OpenCV追従 × 角度採点")

# スマホ画面でスクロールせずにカメラ映像と操作系を1画面に収めるためのCSS
st.markdown(
    """
    <style>
    /* メイン画面内のすべての st.columns をスマホでも横並びに固定（外側の構造用ブロックに影響しないように制限） */
    .block-container [data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
    }
    /* Streamlitがモバイル用（width: 100% !important）に強制するカラムラッパーの幅指定を上書き */
    .block-container [data-testid="stHorizontalBlock"] > div {
        min-width: 0 !important;
        width: auto !important;
        flex: 1 1 0% !important;
    }
    .block-container [data-testid="column"] {
        flex: 1 1 0% !important;
        min-width: 0 !important;
        width: auto !important;
    }
    .block-container [data-testid="stHorizontalBlock"] button {
        padding-left: 2px !important;
        padding-right: 2px !important;
        min-height: 2.2rem !important;
        height: 2.2rem !important;
        font-size: 16px !important;
    }

    /* モバイル（画面幅640px以下）の場合のレイアウト圧縮 */
    @media (max-width: 640px) {
        /* メインコンテナの余白を極限まで削る */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
        }
        /* ヘッダーバーの非表示 */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        /* タイトル・キャプションのサイズ縮小と余白カット */
        h1 {
            font-size: 1.25rem !important;
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-top: 0px !important;
        }
        h2, h3 {
            font-size: 1.05rem !important;
            margin-top: 0.3rem !important;
            margin-bottom: 0.1rem !important;
        }
        .stCaption {
            font-size: 0.75rem !important;
            margin-bottom: 0.2rem !important;
        }
        /* ウィジェット間の不要な余白をカット */
        div[data-testid="stElementContainer"] {
            margin-bottom: 0.15rem !important;
        }
        /* カメラ映像（webrtc_streamer の iframe）の高さを適度なサイズに制限（400pxに調整） */
        .block-container iframe {
            max-height: 400px !important;
            height: 400px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

target_angle = st.sidebar.slider("目標角度 (°)", 10.0, 90.0, 30.0, step=1.0)

# (その他の設定やコントロール類は、スマホ対応のためメイン画面の映像下に配置しました)


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

    # ---------- UI からの設定反映 ----------
    def update_settings(self, mode, tgt_angle, roi_pct, thresh, ai_on,
                        tracking_active, recording, box_positions=None):
        self.mode = mode
        self.target_angle = tgt_angle
        self.roi_percent = roi_pct
        self.threshold = thresh
        self.ai_enabled = ai_on and self.session is not None

        # 枠の位置をスライダーから反映 (中心座標 -> 左上座標に変換)
        # 追従開始後は枠がトラッカーに従うため、選択中(非追従)のみ更新
        if box_positions and not self.tracking_active:
            cx, cy, sz = box_positions["tip"]
            self.box_tip_rel = (cx - sz / 2, cy - sz / 2, sz, sz)
            cx, cy, sz = box_positions["tail"]
            self.box_tail_rel = (cx - sz / 2, cy - sz / 2, sz, sz)

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
            # セットアップ画面 (描画はオリジナル画像に行う)
            cv2.rectangle(img, (tip_x, tip_y),
                          (tip_x + tip_w, tip_y + tip_h), (255, 0, 0), 3)
            cv2.putText(img, "Place TIP here", (tip_x, tip_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.rectangle(img, (tail_x, tail_y),
                          (tail_x + tail_w, tail_y + tail_h), (0, 0, 255), 3)
            cv2.putText(img, "Place BODY here", (tail_x, tail_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "Align Needle & Press 'Lock-on'", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.angle_history.clear()
            return

        # ------------------------------------------------------------
        # 高精度・高速化のための前処理 (解像度最適化 & コントラスト強調)
        # ------------------------------------------------------------
        # 1. 追従処理用の解像度を幅640基準にダウンスケールしてCPU処理負荷を下げ、FPSを向上
        target_w = 640
        scale = target_w / width if width > target_w else 1.0

        # 2. LAB色空間でのコントラスト強調 (CLAHE)
        # 金属製の針やカラーのプラスチック部分が背景の皮膚からくっきり浮き出るようにする
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b_chan))
        img_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        if scale < 1.0:
            img_track = cv2.resize(img_enhanced, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_track = img_enhanced

        # ロックオン開始 (初回のみトラッカー初期化)
        if not self.track_init_done:
            self.tracker_tip = self.create_tracker()
            self.tracker_tail = self.create_tracker()
            if self.tracker_tip and self.tracker_tail:
                # 縮小画像用の位置座標を計算
                tip_box_small = (int(tip_x * scale), int(tip_y * scale), int(tip_w * scale), int(tip_h * scale))
                tail_box_small = (int(tail_x * scale), int(tail_y * scale), int(tail_w * scale), int(tail_h * scale))
                
                self.tracker_tip.init(img_track, tip_box_small)
                self.tracker_tail.init(img_track, tail_box_small)
                self.track_init_done = True

        if not self.track_init_done:
            return

        # 追従実行 (最適化した画像で更新)
        ok_tip, bbox_tip_small = self.tracker_tip.update(img_track)
        ok_tail, bbox_tail_small = self.tracker_tail.update(img_track)

        if ok_tip and ok_tail:
            # 座標を元の高解像度にスケールアップ
            bbox_tip = (int(bbox_tip_small[0] / scale), int(bbox_tip_small[1] / scale),
                        int(bbox_tip_small[2] / scale), int(bbox_tip_small[3] / scale))
            bbox_tail = (int(bbox_tail_small[0] / scale), int(bbox_tail_small[1] / scale),
                         int(bbox_tail_small[2] / scale), int(bbox_tail_small[3] / scale))

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
# 実行環境がローカル（開発PC）か、クラウド（Streamlit Community Cloud等のLinuxコンテナ）かを自動判定。
# ローカル環境では外部STUNへの接続試行によるタイムアウトを防ぐためSTUNを無効化し、
# クラウド環境ではNAT越えのためにGoogleの公開STUNサーバーを有効化します。
import os
is_local = (os.name == 'nt') or not os.path.exists("/home/appuser")

if is_local:
    ice_servers = []
else:
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
# 🎯 操作コントロールパネル（カメラ映像の下に配置し、見ながら操作できるように改善）
# ------------------------------------------------------------
st.markdown("---")

box_positions = None
if mode == "自動検出 (Hough変換)":
    st.subheader("📐 自動検出設定")
    c1, c2 = st.columns(2)
    with c1:
        roi_percent = st.slider("ROIサイズ (%)", 30, 100, 60)
    with c2:
        hsv_threshold = st.slider("HSV彩度閾値", 0, 255, 60)
else:
    if "tracking_active" not in st.session_state:
        st.session_state["tracking_active"] = False
        
    # 枠位置の初期値 (%) — 十字キーで調整
    for _k, _v in (("trk_tip_x", 45), ("trk_tip_y", 30),
                   ("trk_tail_x", 60), ("trk_tail_y", 60), ("trk_box", 12)):
        st.session_state.setdefault(_k, _v)

    if st.session_state.get("tracking_active"):
        st.success("🔒 追従中")
        if st.button("🔄 リセット (枠を合わせ直す)", use_container_width=True, type="primary"):
            st.session_state["tracking_active"] = False
            st.rerun()
    else:
        # 動かす対象の切り替え（テキスト「動かす枠:」と見出しは不要なので削除）
        target_box = st.radio(
            "動かす枠:", ["🔵 針先 (青)", "🔴 根本 (赤)"],
            horizontal=True, key="trk_target",
            label_visibility="collapsed"
        )
            
        prefix = "trk_tip" if "針先" in target_box else "trk_tail"
        STEP = 4

        def _nudge(axis, delta):
            key = f"{prefix}_{axis}"
            st.session_state[key] = int(
                np.clip(st.session_state[key] + delta, 0, 100))

        # 2列のゲームパッド風ボタン配列（スマホ操作用）
        # 1行目: 左・上・右 の移動
        pad_row1 = st.columns(3)
        with pad_row1[0]:
            if st.button("⬅️", use_container_width=True, key="pad_left"):
                _nudge("x", -STEP); st.rerun()
        with pad_row1[1]:
            if st.button("⬆️", use_container_width=True, key="pad_up"):
                _nudge("y", -STEP); st.rerun()
        with pad_row1[2]:
            if st.button("➡️", use_container_width=True, key="pad_right"):
                _nudge("x", STEP); st.rerun()

        # 2行目: 枠縮小・下・枠拡大
        pad_row2 = st.columns(3)
        with pad_row2[0]:
            if st.button("➖", use_container_width=True, key="pad_shrink", help="枠を小さく"):
                st.session_state["trk_box"] = int(np.clip(st.session_state["trk_box"] - 2, 5, 30))
                st.rerun()
        with pad_row2[1]:
            if st.button("⬇️", use_container_width=True, key="pad_down"):
                _nudge("y", STEP); st.rerun()
        with pad_row2[2]:
            if st.button("➕", use_container_width=True, key="pad_grow", help="枠を大きく"):
                st.session_state["trk_box"] = int(np.clip(st.session_state["trk_box"] + 2, 5, 30))
                st.rerun()

        st.caption(
            f"🔵 ({st.session_state['trk_tip_x']},{st.session_state['trk_tip_y']}) | "
            f"🔴 ({st.session_state['trk_tail_x']},{st.session_state['trk_tail_y']}) | "
            f"枠幅 {st.session_state['trk_box']}%"
        )

        if st.button("🎯 ロックオン (追従開始)", type="primary", use_container_width=True):
            st.session_state["tracking_active"] = True
            st.rerun()

    # 現在の枠位置を box_positions にまとめる (中心座標)
    box_positions = {
        "tip": (st.session_state["trk_tip_x"] / 100,
                st.session_state["trk_tip_y"] / 100,
                st.session_state["trk_box"] / 100),
        "tail": (st.session_state["trk_tail_x"] / 100,
                 st.session_state["trk_tail_y"] / 100,
                 st.session_state["trk_box"] / 100),
    }

    roi_percent = 60
    hsv_threshold = 60

# テスト（採点）の開始・終了コントロール
if "is_recording" not in st.session_state:
    st.session_state["is_recording"] = False

rec_col1, rec_col2 = st.columns(2)
with rec_col1:
    btn_type = "secondary" if st.session_state["is_recording"] else "primary"
    start_test = st.button("▶️ テスト開始", use_container_width=True, type=btn_type)
with rec_col2:
    stop_test = st.button("⏹ 終了・採点", use_container_width=True, disabled=not st.session_state["is_recording"])

# ------------------------------------------------------------
# テスト (採点) 制御と結果表示
# ------------------------------------------------------------
if start_test:
    st.session_state["is_recording"] = True
    st.session_state.pop("test_result", None)
    st.rerun()
if stop_test and st.session_state["is_recording"]:
    st.session_state["is_recording"] = False
    if ctx.video_processor:
        with ctx.video_processor.lock:
            angles = list(ctx.video_processor.recorded_angles)
        st.session_state["test_result"] = angles
    st.rerun()

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
        box_positions=box_positions if proc_mode == "Tracking" else None,
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
