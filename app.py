import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- UI設定 ---
st.set_page_config(page_title="穿刺ガイドシミュレータ", layout="wide")
st.title("💉 穿刺ガイドシミュレータ (クラウド対応版)")

# --- サイドバー設定 ---

# 1. カメラ切り替え
st.sidebar.header("📷 カメラ設定")
camera_mode = st.sidebar.radio(
    "カメラの向き",
    ("インカメラ (自分側)", "アウトカメラ (外側)"),
    index=1
)

if camera_mode == "インカメラ (自分側)":
    video_constraints = {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}
else:
    # スマホのアウトカメラ用設定
    video_constraints = {"facingMode": "environment", "width": {"ideal": 640}, "height": {"ideal": 480}}

# 2. ガイド機能
st.sidebar.markdown("---")
st.sidebar.header("📏 ガイド設定")
show_guide = st.sidebar.checkbox("疑似針（ガイド線）を表示", value=True)
guide_length_mm = st.sidebar.slider("疑似針の長さ (mm)", 1.0, 5.0, 3.0, step=0.5)

# 3. 調整用
st.sidebar.markdown("---")
st.sidebar.subheader("👀 調整")
show_edge = st.sidebar.checkbox("エッジのみ表示 (認識確認)", value=False)


# --- Logic: 映像処理クラス ---
class NeedleGuideSimulator(VideoProcessorBase):
    def __init__(self):
        self.show_guide = True
        self.guide_len_mm = 3.0
        self.debug_mode = False
        self.PX_PER_MM = 20.0 

    def update_settings(self, guide_on, guide_len_mm, debug):
        self.show_guide = guide_on
        self.guide_len_mm = guide_len_mm
        self.debug_mode = debug

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # 画像処理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            if self.debug_mode:
                return av.VideoFrame.from_ndarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), format="bgr24")

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=60, maxLineGap=20)

            if lines is not None:
                best_line = None
                max_len = 0
                current_angle = 0.0
                
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    if lx2 - lx1 == 0: la = 90.0
                    else: la = np.degrees(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1)))
                    
                    if 10 < la < 85:
                        length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                        if length > max_len:
                            max_len = length
                            best_line = line
                            current_angle = la
                
                if best_line is not None:
                    bx1, by1, bx2, by2 = best_line[0]
                    if by1 > by2: 
                        tip = (bx1, by1); tail = (bx2, by2)
                    else: 
                        tip = (bx2, by2); tail = (bx1, by1)
                    
                    status_color = (0, 255, 255) 
                    if 20 <= current_angle <= 40:
                        status_color = (255, 100, 0) 
                    
                    cv2.line(img, tail, tip, status_color, 6)
                    
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
                            cv2.circle(img, (guide_end_x, guide_end_y), 3, (255, 255, 0), -1)

                    msg = f"Angle: {current_angle:.1f}"
                    cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            # エラー時に真っ黒にならないように
            return frame

# --- メイン実行部 (ここが重要！) ---
# クラウド環境で繋がりやすくするためのサーバーリスト
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
}

ctx = webrtc_streamer(
    key="needle-cloud-mode",
    video_processor_factory=NeedleGuideSimulator,
    rtc_configuration=RTC_CONFIGURATION, # 強化した設定を使用
    media_stream_constraints={"video": video_constraints, "audio": False}
)

if ctx.video_processor:
    ctx.video_processor.update_settings(show_guide, guide_length_mm, show_edge)
