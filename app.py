import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- UI設定 ---
st.set_page_config(page_title="穿刺シミュレータ(安定版)", layout="wide")
st.title("💉 穿刺シミュレータ (基本機能・安定版)")

# --- サイドバー設定 ---
st.sidebar.header("🔧 血管の位置調整")
s_pos_x = st.sidebar.slider("横位置 (X)", 0, 640, 320, step=10)
s_pos_y = st.sidebar.slider("深さ (Y)", 0, 480, 300, step=10)
s_angle = st.sidebar.slider("傾き", -45, 45, 0, step=1)
s_diam = st.sidebar.select_slider("血管径 (mm)", options=[4, 5, 6], value=5)

st.sidebar.markdown("---")
st.sidebar.header("👀 調整モード")
# これをONにすると、カメラが「どう見えているか」が白黒でわかります
show_edge_view = st.sidebar.checkbox("輪郭(エッジ)だけを表示する", value=False)
st.sidebar.info("針が認識されない時は、上のチェックを入れてください。「針の形」が白く浮き出ていなければ、照明や背景を調整する必要があります。")

# --- Logic: 映像処理クラス ---
class PenetrationSimulator(VideoProcessorBase):
    def __init__(self):
        # 初期値
        self.vessel_x = 320
        self.vessel_y = 300
        self.vessel_angle = 0
        self.vessel_d_mm = 5
        self.debug_mode = False # エッジ表示モード

    def update_settings(self, x, y, angle, d_mm, debug):
        self.vessel_x = x
        self.vessel_y = y
        self.vessel_angle = angle
        self.vessel_d_mm = d_mm
        self.debug_mode = debug

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # --- 1. 血管の描画（常に表示） ---
            # クラス内の変数を使用
            diameter_px = self.vessel_d_mm * 5 
            length_px = 800
            
            rad = np.radians(self.vessel_angle)
            dx = np.cos(rad)
            dy = np.sin(rad)
            
            cx, cy = self.vessel_x, self.vessel_y
            x1 = int(cx - dx * length_px/2)
            y1 = int(cy - dy * length_px/2)
            x2 = int(cx + dx * length_px/2)
            y2 = int(cy + dy * length_px/2)
            
            ox = -dy * diameter_px
            oy = dx * diameter_px
            
            p_top1 = (int(x1 + ox), int(y1 + oy))
            p_top2 = (int(x2 + ox), int(y2 + oy))
            p_bot1 = (int(x1 - ox), int(y1 - oy))
            p_bot2 = (int(x2 - ox), int(y2 - oy))
            
            # 血管エリアのY座標（簡易判定用）
            vessel_top_y = min(p_top1[1], p_top2[1])
            vessel_bot_y = max(p_bot1[1], p_bot2[1])

            # --- 2. 針の検出処理 ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 以前調子が良かった設定値に戻します (50, 150)
            edges = cv2.Canny(blurred, 50, 150)
            
            # もし「輪郭だけ表示モード」なら、ここで画像を差し替えて終了
            if self.debug_mode:
                # 血管の線だけエッジ画像に書き足してあげる（位置合わせ用）
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cv2.line(edges_bgr, p_top1, p_top2, (0, 0, 255), 2)
                cv2.line(edges_bgr, p_bot1, p_bot2, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(edges_bgr, format="bgr24")

            # 直線検出 (パラメータを標準的で少し緩めに設定)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=50,      # 80->50に下げて検出しやすく
                                    minLineLength=60,  # 100->60に下げて短い針も拾う
                                    maxLineGap=20)
            
            # 血管描画 (カラー画像用)
            cv2.line(img, p_top1, p_top2, (0, 0, 200), 2)
            cv2.line(img, p_bot1, p_bot2, (0, 0, 150), 2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)

            if lines is not None:
                best_line = None
                max_len = 0
                current_angle = 0.0
                
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    # 角度計算
                    if lx2 - lx1 == 0: la = 90.0
                    else: la = np.degrees(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1)))
                    
                    # 角度フィルタ (水平・垂直すぎる線は無視)
                    if 10 < la < 85:
                        length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                        if length > max_len:
                            max_len = length
                            best_line = line
                            current_angle = la
                
                if best_line is not None:
                    bx1, by1, bx2, by2 = best_line[0]
                    if by1 > by2: tip = (bx1, by1); tail = (bx2, by2)
                    else: tip = (bx2, by2); tail = (bx1, by1)
                    
                    # 判定ロジック
                    status_color = (0, 255, 255) # 黄
                    msg = f"Angle: {current_angle:.1f}"

                    if 20 <= current_angle <= 40:
                        status_color = (255, 100, 0) # 青(OK)
                    
                    # 血管判定
                    if tip[1] > vessel_top_y:
                        msg = "IN VESSEL"
                        status_color = (0, 255, 0) # 緑
                        if tip[1] > vessel_bot_y:
                            msg = "PENETRATION!!"
                            status_color = (0, 0, 255) # 赤
                            cv2.rectangle(img, (0,0), (640,480), (0,0,255), 5)
                    
                    cv2.line(img, tail, tip, status_color, 6)
                    cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            err_img = frame.to_ndarray(format="bgr24")
            cv2.putText(err_img, f"Error: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(err_img, format="bgr24")

# --- メイン実行部 ---
ctx = webrtc_streamer(
    key="stable-mode",
    video_processor_factory=PenetrationSimulator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

if ctx.video_processor:
    ctx.video_processor.update_settings(s_pos_x, s_pos_y, s_angle, s_diam, show_edge_view)