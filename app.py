import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- UI設定 ---
st.set_page_config(page_title="穿刺VRシミュレータ", layout="wide")
st.title("💉 穿刺VRシミュレータ (スマホ対応版)")

# --- サイドバー設定 ---

# 1. カメラ切り替え設定
st.sidebar.header("📷 カメラ設定")
camera_mode = st.sidebar.radio(
    "カメラの向き",
    ("インカメラ (自分側)", "アウトカメラ (外側)"),
    index=1 # デフォルトはアウトカメラ（患者撮影用）
)

# video_constraintsの設定（スマホのカメラ切り替え用）
if camera_mode == "インカメラ (自分側)":
    # facingMode: "user" はインカメラ
    video_constraints = {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}
else:
    # facingMode: "environment" はアウトカメラ
    video_constraints = {"facingMode": "environment", "width": {"ideal": 640}, "height": {"ideal": 480}}

# 2. シミュレーション条件
st.sidebar.markdown("---")
st.sidebar.header("📏 条件設定")
st.sidebar.warning("⚠️ **前提条件: カメラと穿刺部位の距離を「約30cm」に保ってください。**")

s_diam = st.sidebar.select_slider("血管径 (mm)", options=[4, 5, 6], value=5)

# 3. 血管の位置調整（VR空間座標）
st.sidebar.subheader("VR血管の位置調整")
st.sidebar.info("30cm先にあると仮定して描画します。画面を見ながら血管位置を微調整してください。")
s_pos_x = st.sidebar.slider("横位置 (X)", -150, 150, 0, step=5, help="中心からのズレ(mm)")
s_pos_y = st.sidebar.slider("縦位置 (Y)", -100, 100, 0, step=5, help="中心からのズレ(mm)")
s_angle = st.sidebar.slider("血管の傾き", -45, 45, 0, step=1)

# 4. デバッグ
show_edge = st.sidebar.checkbox("エッジのみ表示 (認識確認用)", value=False)


# --- Logic: 映像処理クラス ---
class VRPenetrationSimulator(VideoProcessorBase):
    def __init__(self):
        # 3D空間上の血管位置（初期値）
        self.offset_x = 0
        self.offset_y = 0
        self.angle = 0
        self.diameter_mm = 5
        self.debug_mode = False
        
        # 固定パラメータ（距離30cm）
        self.DISTANCE_MM = 300.0 

    def update_settings(self, x, y, angle, d_mm, debug):
        self.offset_x = x
        self.offset_y = y
        self.angle = angle
        self.diameter_mm = d_mm
        self.debug_mode = debug

    def draw_vr_vessel(self, img):
        h, w, c = img.shape
        
        # 簡易カメラ行列（スマホの一般的な画角を想定）
        # 焦点距離 f は、画角60度くらいと仮定すると、横幅(w)と同程度になります
        focal_length = w 
        cam_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1)) # 歪みなし

        # 血管（円柱）の定義
        radius = self.diameter_mm / 2.0
        length_mm = 120.0 # 血管の長さ
        
        # 3D空間での座標定義
        # カメラからZ軸方向に300mm離れた場所を基準にする
        # 回転を考慮して、始点と終点を計算
        rad = np.radians(self.angle)
        dx = np.cos(rad) * (length_mm / 2)
        dy = np.sin(rad) * (length_mm / 2)
        
        # 3D座標 (X, Y, Z)
        # X: サイドバー調整 + 左右への広がり
        # Y: サイドバー調整 + 上下への広がり
        # Z: 常に300mm (固定)
        
        # 中心線
        p_start_3d = np.array([[self.offset_x - dx, self.offset_y - dy, self.DISTANCE_MM]], dtype=np.float32)
        p_end_3d = np.array([[self.offset_x + dx, self.offset_y + dy, self.DISTANCE_MM]], dtype=np.float32)
        
        # 上壁と下壁（Y軸方向にずらす簡易計算）
        # ※本来は回転に合わせて法線ベクトルを計算すべきですが、簡易的にY軸シフトで表現
        y_shift_x = -np.sin(rad) * radius
        y_shift_y = np.cos(rad) * radius
        
        wall_top_start = p_start_3d + np.array([y_shift_x, y_shift_y, 0])
        wall_top_end   = p_end_3d   + np.array([y_shift_x, y_shift_y, 0])
        wall_bot_start = p_start_3d - np.array([y_shift_x, y_shift_y, 0])
        wall_bot_end   = p_end_3d   - np.array([y_shift_x, y_shift_y, 0])

        # 3D -> 2D 投影 (ProjectPoints)
        # 回転・並進ベクトルは0（座標自体を動かしたので）
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

        p_s_2d, _ = cv2.projectPoints(p_start_3d, rvec, tvec, cam_matrix, dist_coeffs)
        p_e_2d, _ = cv2.projectPoints(p_end_3d, rvec, tvec, cam_matrix, dist_coeffs)
        
        wt_s_2d, _ = cv2.projectPoints(wall_top_start, rvec, tvec, cam_matrix, dist_coeffs)
        wt_e_2d, _ = cv2.projectPoints(wall_top_end,   rvec, tvec, cam_matrix, dist_coeffs)
        wb_s_2d, _ = cv2.projectPoints(wall_bot_start, rvec, tvec, cam_matrix, dist_coeffs)
        wb_e_2d, _ = cv2.projectPoints(wall_bot_end,   rvec, tvec, cam_matrix, dist_coeffs)

        # 整数座標に変換
        def to_pt(cv_point): return tuple(np.int32(cv_point).reshape(2))
        
        ps, pe = to_pt(p_s_2d), to_pt(p_e_2d)
        wts, wte = to_pt(wt_s_2d), to_pt(wt_e_2d)
        wbs, wbe = to_pt(wb_s_2d), to_pt(wb_e_2d)

        # 描画
        # 血管の壁 (赤)
        cv2.line(img, wts, wte, (0, 0, 200), 2)
        cv2.line(img, wbs, wbe, (0, 0, 150), 2)
        # 中心線 (黄色)
        cv2.line(img, ps, pe, (0, 255, 255), 1)
        
        # 始点と終点の円（それっぽく見せる装飾）
        cv2.line(img, wts, wbs, (0, 0, 200), 1)
        cv2.line(img, wte, wbe, (0, 0, 200), 1)
        
        cv2.putText(img, f"Virtual Vessel ({self.diameter_mm}mm)", (wts[0], wts[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # 判定用のY座標（画面上の平均的な高さ）を返す
        vessel_top_y = (wts[1] + wte[1]) / 2
        vessel_bot_y = (wbs[1] + wbe[1]) / 2
        
        return vessel_top_y, vessel_bot_y


    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # --- 1. VR血管描画 ---
            # 距離30cmを想定した3D投影で描画します
            v_top, v_bot = self.draw_vr_vessel(img)

            # --- 2. 針の検出 ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            if self.debug_mode:
                # エッジ確認モードならここでリターン
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
                    
                    # 角度フィルタ (水平・垂直すぎる線は無視)
                    if 10 < la < 85:
                        length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                        if length > max_len:
                            max_len = length
                            best_line = line
                            current_angle = la
                
                if best_line is not None:
                    bx1, by1, bx2, by2 = best_line[0]
                    # 先端(下)を特定
                    if by1 > by2: tip = (bx1, by1); tail = (bx2, by2)
                    else: tip = (bx2, by2); tail = (bx1, by1)
                    
                    # 判定ロジック
                    status_color = (0, 255, 255) # 黄
                    msg = f"Angle: {current_angle:.1f}"

                    if 20 <= current_angle <= 40:
                        status_color = (255, 100, 0) # 青(OK)
                    
                    # 貫通判定 (Y座標ベース)
                    if tip[1] > v_top:
                        msg = "IN VESSEL"
                        status_color = (0, 255, 0) # 緑
                        if tip[1] > v_bot:
                            msg = "PENETRATION!!"
                            status_color = (0, 0, 255) # 赤
                            h, w, _ = img.shape
                            cv2.rectangle(img, (0,0), (w, h), (0, 0, 255), 5)
                    
                    cv2.line(img, tail, tip, status_color, 6)
                    cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            # エラー時
            err_img = frame.to_ndarray(format="bgr24")
            cv2.putText(err_img, f"Error: {e}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(err_img, format="bgr24")

# --- メイン実行部 ---
ctx = webrtc_streamer(
    key="vr-mobile-mode",
    video_processor_factory=VRPenetrationSimulator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    # スマホのカメラ切り替え設定を反映
    media_stream_constraints={"video": video_constraints, "audio": False}
)

if ctx.video_processor:
    ctx.video_processor.update_settings(s_pos_x, s_pos_y, s_angle, s_diam, show_edge)
