import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- UI設定 ---
st.set_page_config(page_title="穿刺ガイドシミュレータ", layout="wide")
st.title("💉 穿刺ガイドシミュレータ (ガイド線機能付)")

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
show_guide = st.sidebar.checkbox("疑似針（ガイド線）を表示", value=True, help="針の延長線を表示して、刺入位置を予測します")
guide_length = st.sidebar.slider("ガイド線の長さ", 100, 1000, 500, step=50)

# 3. 調整用（認識されにくい時用）
st.sidebar.markdown("---")
st.sidebar.subheader("👀 調整")
show_edge = st.sidebar.checkbox("エッジのみ表示 (認識確認)", value=False)


# --- Logic: 映像処理クラス ---
class NeedleGuideSimulator(VideoProcessorBase):
    def __init__(self):
        self.show_guide = True
        self.guide_len = 500
        self.debug_mode = False

    def update_settings(self, guide_on, guide_len, debug):
        self.show_guide = guide_on
        self.guide_len = guide_len
        self.debug_mode = debug

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # 1. 針の検出処理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # デバッグモード：エッジだけ返す
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
                    
                    # 角度フィルタ (10度〜85度)
                    if 10 < la < 85:
                        length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                        if length > max_len:
                            max_len = length
                            best_line = line
                            current_angle = la
                
                # 針が見つかった場合の描画処理
                if best_line is not None:
                    bx1, by1, bx2, by2 = best_line[0]
                    # 先端(Y座標が大きい方＝下)を特定
                    if by1 > by2: 
                        tip = (bx1, by1)  # 先端
                        tail = (bx2, by2) # 根元
                    else: 
                        tip = (bx2, by2)
                        tail = (bx1, by1)
                    
                    # 角度による色分け
                    status_color = (0, 255, 255) # 黄 (注意)
                    if 20 <= current_angle <= 40:
                        status_color = (255, 100, 0) # 青 (良好)
                    
                    # --- 針本体の描画 ---
                    cv2.line(img, tail, tip, status_color, 6)
                    
                    # --- 疑似針（ガイド線）の描画 ---
                    if self.show_guide:
                        # 単位ベクトルを計算
                        vec_x = tip[0] - tail[0]
                        vec_y = tip[1] - tail[1]
                        vec_len = np.sqrt(vec_x**2 + vec_y**2)
                        
                        if vec_len > 0:
                            unit_x = vec_x / vec_len
                            unit_y = vec_y / vec_len
                            
                            # ガイドの終点計算
                            guide_end_x = int(tip[0] + unit_x * self.guide_len)
                            guide_end_y = int(tip[1] + unit_y * self.guide_len)
                            
                            # ガイド線を描画（水色）
                            cv2.line(img, tip, (guide_end_x, guide_end_y), (255, 255, 0), 2)
                            # 先端にマーク
                            cv2.circle(img, tip, 5, (255, 255, 0), -1)

                    # テキスト情報
                    msg = f"Angle: {current_angle:.1f}"
                    cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            err_img = frame.to_ndarray(format="bgr24")
            cv2.putText(err_img, f"Error: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(err_img, format="bgr24")

# --- メイン実行部 ---
ctx = webrtc_streamer(
    key="needle-guide-mode",
    video_processor_factory=NeedleGuideSimulator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": video_constraints, "audio": False}
)

if ctx.video_processor:
    ctx.video_processor.update_settings(show_guide, guide_length, show_edge)
