import streamlit as st
import cv2
import numpy as np
import av
# WebRtcMode を追加インポート
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- UI設定 ---
st.set_page_config(page_title="穿刺ガイドシミュレータ", layout="wide")
st.title("💉 穿刺ガイドシミュレータ (クラウド対応版)")
st.caption("Ver 1.1 - Fixed Mode & Connection")

# --- サイドバー設定 ---

# 1. カメラ切り替え
st.sidebar.header("📷 カメラ設定")
camera_mode = st.sidebar.radio(
    "カメラの向き",
    ("インカメラ (自分側)", "アウトカメラ (外側)"),
    index=1
)

if camera_mode == "インカメラ (自分側)":
    # PCやインカメラ用
    video_constraints = {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}
else:
    # スマホのアウトカメラ用（environment）
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
        self.PX_PER_MM = 20.0  # ※仮定値: 実際の距離校正は別途必要

    def update_settings(self, guide_on, guide_len_mm, debug):
        self.show_guide = guide_on
        self.guide_len_mm = guide_len_mm
        self.debug_mode = debug

    def recv(self, frame):
        try:
            # WebRTCのフレームをnumpy配列(BGR)に変換
            img = frame.to_ndarray(format="bgr24")
            
            # --- 画像処理プロセス ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # デバッグモードならエッジ画像だけを返す
            if self.debug_mode:
                return av.VideoFrame.from_ndarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), format="bgr24")

            # 直線検出 (Hough変換)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=60, maxLineGap=20)

            best_line = None
            max_len = 0
            current_angle = 0.0
            
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    
                    # 角度計算 (0除算回避)
                    if lx2 - lx1 == 0: 
                        la = 90.0
                    else: 
                        la = np.degrees(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1)))
                    
                    # 穿刺角度としてあり得る範囲(10度〜85度)の線だけ採用
                    if 10 < la < 85:
                        length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                        # 最も長い線を「針」とみなす
                        if length > max_len:
                            max_len = length
                            best_line = line
                            current_angle = la
                
                # ベストな線が見つかった場合の描画処理
                if best_line is not None:
                    bx1, by1, bx2, by2 = best_line[0]
                    
                    # 針先判定: 画面の下側(yが大きい方)を根本、上側を針先と仮定する簡易ロジック
                    if by1 < by2: 
                        tip = (bx1, by1); tail = (bx2, by2)
                    else: 
                        tip = (bx2, by2); tail = (bx1, by1)
                    
                    # 角度による色分け (例: 20-40度が推奨範囲ならオレンジ、それ以外は黄色)
                    status_color = (0, 255, 255) # Yellow
                    if 20 <= current_angle <= 40:
                        status_color = (0, 165, 255) # Orange (BGR)
                    
                    # 実線の描画
                    cv2.line(img, tail, tip, status_color, 6)
                    
                    # ガイド線(延長線)の描画
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
                            
                            # ガイド線描画
                            cv2.line(img, tip, (guide_end_x, guide_end_y), (255, 255, 0), 3)
                            # 先端の点
                            cv2.circle(img, (guide_end_x, guide_end_y), 5, (255, 255, 0), -1)

                    # 角度表示
                    msg = f"Angle: {current_angle:.1f}"
                    cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            # エラーが起きてもストリームを止めない（コンソールには出す）
            print(f"Error processing frame: {e}")
            return frame

# --- メイン実行部 ---

# クラウド環境用の強力なSTUN設定
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Streamlit UIへの配置
ctx = webrtc_streamer(
    key="needle-cloud-mode",
    mode=WebRtcMode.SENDRECV, # <--- 【重要修正】文字列ではなくEnumを使用
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=NeedleGuideSimulator,
    media_stream_constraints={"video": video_constraints, "audio": False},
    async_processing=True,
)

# パラメータ動的反映
if ctx.video_processor:
    ctx.video_processor.update_settings(show_guide, guide_length_mm, show_edge)
