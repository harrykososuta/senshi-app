import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- UI設定 ---
st.set_page_config(page_title="穿刺ガイドシミュレータ", layout="wide")
st.title("💉 穿刺ガイド - フォーカスモード搭載")
st.caption("Ver 4.0 - ROI Focus")

# --- 通信設定 ---
# ※ここに前回の Metered.ca の設定（TURN_USERNAME, TURN_PASSWORD）があればそのまま使ってください
# なければGoogleの無料サーバーを使います（Wi-Fi推奨）
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- サイドバー設定 ---
st.sidebar.header("📷 カメラ設定")
camera_mode = st.sidebar.radio(
    "カメラの向き",
    ("インカメラ (自分側)", "アウトカメラ (外側)"),
    index=1
)

if camera_mode == "インカメラ (自分側)":
    video_constraints = {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}
else:
    video_constraints = {"facingMode": "environment", "width": {"ideal": 640}, "height": {"ideal": 480}}

st.sidebar.markdown("---")
st.sidebar.header("🎯 認識範囲の設定")
st.sidebar.info("背景の誤検知を防ぐため、認識する範囲を絞ります。")
roi_size = st.sidebar.slider("検出枠のサイズ (%)", 10, 100, 50, help="値を小さくすると、画面中央のみを解析します")

st.sidebar.markdown("---")
st.sidebar.header("📏 ガイド設定")
show_guide = st.sidebar.checkbox("疑似針（ガイド線）を表示", value=True)
guide_length_mm = st.sidebar.slider("疑似針の長さ (mm)", 1.0, 5.0, 3.0, step=0.5)
show_debug = st.sidebar.checkbox("解析領域を確認 (デバッグ)", value=False)

# --- 映像処理クラス ---
class NeedleGuideSimulator(VideoProcessorBase):
    def __init__(self):
        self.show_guide = True
        self.guide_len_mm = 3.0
        self.roi_percent = 50
        self.show_debug = False
        self.PX_PER_MM = 20.0 

    def update_settings(self, guide_on, guide_len_mm, roi, debug):
        self.show_guide = guide_on
        self.guide_len_mm = guide_len_mm
        self.roi_percent = roi
        self.show_debug = debug

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            height, width = img.shape[:2]

            # --- 1. ROI（注目エリア）の計算 ---
            # 画面中央から指定された％分の領域を計算
            roi_w = int(width * (self.roi_percent / 100))
            roi_h = int(height * (self.roi_percent / 100))
            roi_x = int((width - roi_w) / 2)
            roi_y = int((height - roi_h) / 2)

            # --- 2. 解析用の画像を作成 ---
            # まず真っ黒な画像を作る
            mask = np.zeros_like(img)
            # 注目エリアだけ白い四角を描く
            cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), -1)
            
            # 元画像とマスクを合成（注目エリア以外を黒く塗りつぶした画像を作る）
            masked_img = cv2.bitwise_and(img, mask)

            # --- 3. 画像処理（マスクされた画像に対して行う） ---
            gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # デバッグモード：認識しているエッジを表示
            if self.show_debug:
                # 枠を描画して返す
                cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                # エッジ画像をカラー変換して合成（透かして見せる）
                edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                return av.VideoFrame.from_ndarray(cv2.addWeighted(img, 0.8, edge_color, 0.5, 0), format="bgr24")

            # 直線検出
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=60, maxLineGap=20)
            
            best_line = None
            max_len = 0
            current_angle = 0.0
            
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    
                    # 角度計算
                    if lx2 - lx1 == 0: la = 90.0
                    else: la = np.degrees(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1)))
                    
                    # 角度フィルタ（極端な角度は除外）
                    if 10 < la < 85:
                        length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                        if length > max_len:
                            max_len = length
                            best_line = line
                            current_angle = la
                
                # --- 4. 描画（元の綺麗な画像の上に描く） ---
                if best_line is not None:
                    bx1, by1, bx2, by2 = best_line[0]
                    if by1 < by2: tip = (bx1, by1); tail = (bx2, by2)
                    else: tip = (bx2, by2); tail = (bx1, by1)
                    
                    status_color = (0, 255, 255)
                    if 20 <= current_angle <= 40: status_color = (0, 165, 255)
                    
                    # 針の線
                    cv2.line(img, tail, tip, status_color, 6)
                    
                    # ガイド線
                    if self.show_guide:
                        vec_x = tip[0] - tail[0]
                        vec_y = tip[1] - tail[1]
                        vec_len = np.sqrt(vec_x**2 + vec_y**2)
                        if vec_len > 0:
                            unit_x = vec_x / vec_len
                            unit_y = vec_y / vec_len
                            pixel_length = self.guide_len_mm * self.PX_PER_MM
                            guide_end_x = int(tip[0] + unit_x * pixel_length)
                            guide_end_y = int(tip[1] + unit_y * pixel_length)
                            cv2.line(img, tip, (guide_end_x, guide_end_y), (255, 255, 0), 3)
                            cv2.circle(img, (guide_end_x, guide_end_y), 5, (255, 255, 0), -1)

                    msg = f"Angle: {current_angle:.1f}"
                    cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # ユーザーへの案内用に、認識エリア（ROI）の枠を薄く表示する
            cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            cv2.putText(img, "Target Area", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Error: {e}")
            return frame

# --- メイン実行部 ---
ctx = webrtc_streamer(
    key="needle-roi-focus",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=NeedleGuideSimulator,
    media_stream_constraints={"video": video_constraints, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.update_settings(show_guide, guide_length_mm, roi_size, show_debug)
