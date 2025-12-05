import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- UI設定 ---
st.set_page_config(page_title="穿刺ガイドシミュレータ", layout="wide")
st.title("💉 穿刺ガイドシミュレータ (mm指定版)")

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
    video_constraints = {"facingMode": "environment", "width": {"ideal": 640}, "height": {"ideal": 480}}

# 2. ガイド機能
st.sidebar.markdown("---")
st.sidebar.header("📏 ガイド設定")
show_guide = st.sidebar.checkbox("疑似針（ガイド線）を表示", value=True)

# 長さを 1mm ～ 5mm に変更
# step=0.5 にしているので、1.0, 1.5, ... 5.0mm まで調整可能です
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
        
        # 【重要】1mmが何ピクセルか？の定義
        # カメラの距離によりますが、接写(マクロ)と仮定して大きめに設定します
        self.PX_PER_MM = 20.0 

    def update_settings(self, guide_on, guide_len_mm, debug):
        self.show_guide = guide_on
        self.guide_len_mm = guide_len_mm
        self.debug_mode = debug

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # 1. 針の検出処理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # デバッグモード
            if self.debug_mode:
                return av.VideoFrame.from_ndarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), format="bgr24")

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=60, maxLineGap=20)

            if lines is not None:
                best_line = None
                max_len = 0
                current_angle = 0.0
                
                # 最も確からしい針を探す
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
                
                # 描画処理
                if best_line is not None:
                    bx1, by1, bx2, by2 = best_line[0]
                    # 先端(Y座標が大きい方＝下)を特定
                    if by1 > by2: 
                        tip = (bx1, by1); tail = (bx2, by2)
                    else: 
                        tip = (bx2, by2); tail = (bx1, by1)
                    
                    status_color = (0, 255, 255) # 黄
                    if 20 <= current_angle <= 40:
                        status_color = (255, 100, 0) # 青
                    
                    # 針本体
                    cv2.line(img, tail, tip, status_color, 6)
                    
                    # --- 疑似針（mm指定）の描画 ---
                    if self.show_guide:
                        vec_x = tip[0] - tail[0]
                        vec_y = tip[1] - tail[1]
                        vec_len = np.sqrt(vec_x**2 + vec_y**2)
                        
                        if vec_len > 0:
                            unit_x = vec_x / vec_len
                            unit_y = vec_y / vec_len
                            
                            # mm を px に変換して長さを決定
                            pixel_length = self.guide_len_mm * self.PX_PER_MM
                            
                            guide_end_x = int(tip[0] + unit_x * pixel_length)
                            guide_end_y = int(tip[1] + unit_y * pixel_length)
                            
                            # ガイド線（水色）
                            cv2.line(img, tip, (guide_end_x, guide_end_y), (255, 255, 0), 3)
                            # 先端に小さな点
                            cv2.circle(img, (guide_end_x, guide_end_y), 3, (255, 255, 0), -1)

                    # テキスト
                    msg = f"Angle: {current_angle:.1f}"
                    cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            err_img = frame.to_ndarray(format="bgr24")
            cv2.putText(err_img, f"Error: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(err_img, format="bgr24")

# --- メイン実行部 ---
ctx = webrtc_streamer(
    key="needle-mm-guide",
    video_processor_factory=NeedleGuideSimulator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": video_constraints, "audio": False}
)

if ctx.video_processor:
    ctx.video_processor.update_settings(show_guide, guide_length_mm, show_edge)
