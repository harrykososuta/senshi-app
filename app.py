import streamlit as st
import cv2
import numpy as np
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- UI設定 ---
st.set_page_config(page_title="穿刺ガイド - 実践モード", layout="centered") # スマホで見やすいようcenteredに変更
st.title("💉 穿刺ガイド - 実践テストモード")

# --- 通信設定 ---
TURN_USERNAME = "【ここにusername】"
TURN_PASSWORD = "【ここにpassword】"

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # Metered設定がある場合はコメントアウトを外す
            # {
            #     "urls": ["turn:global.turn.metered.ca:80", "turn:global.turn.metered.ca:443"],
            #     "username": TURN_USERNAME,
            #     "credential": TURN_PASSWORD,
            # },
        ]
    }
)

# --- サイドバー設定（調整項目のみ） ---
st.sidebar.header("⚙️ 調整")
st.sidebar.subheader("🎥 認識設定")
roi_size = st.sidebar.slider("検出枠サイズ (%)", 10, 100, 40)
threshold = st.sidebar.slider("検出感度", 30, 150, 50)
flip_tip = st.sidebar.checkbox("針先の向きを反転", value=False, help="ガイド線が逆に出る場合はチェックしてください")

st.sidebar.subheader("🧪 テスト基準")
target_angle = st.sidebar.number_input("目標角度 (度)", 10.0, 60.0, 30.0, step=1.0)
guide_len_mm = st.sidebar.slider("ガイド線の長さ (mm)", 1.0, 10.0, 5.0, step=0.5)

# --- 映像処理クラス ---
class NeedleGuideSimulator(VideoProcessorBase):
    def __init__(self):
        self.roi_percent = 40
        self.threshold = 50
        self.target_angle = 30.0
        self.flip_tip = False
        self.guide_len_mm = 5.0
        
        self.is_recording = False
        self.angle_history = []
        self.last_frame = None
        self.PX_PER_MM = 20.0 

    def update_settings(self, roi, thresh, target, flip, guide_len):
        self.roi_percent = roi
        self.threshold = thresh
        self.target_angle = target
        self.flip_tip = flip
        self.guide_len_mm = guide_len

    def start_test(self):
        self.angle_history = []
        self.is_recording = True

    def stop_test(self):
        self.is_recording = False
        return self.angle_history

    def get_last_frame(self):
        return self.last_frame

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            height, width = img.shape[:2]
            
            # --- ROI計算 ---
            roi_w = int(width * (self.roi_percent / 100))
            roi_h = int(height * (self.roi_percent / 100))
            roi_x = int((width - roi_w) / 2)
            roi_y = int((height - roi_h) / 2)

            mask = np.zeros_like(img)
            cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), -1)
            masked_img = cv2.bitwise_and(img, mask)

            gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.threshold, minLineLength=60, maxLineGap=20)
            
            current_angle = None
            best_line = None
            max_len = 0
            
            if lines is not None:
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

            # --- 描画 ---
            if best_line is not None:
                bx1, by1, bx2, by2 = best_line[0]
                
                # 針先(Tip)の判定ロジック修正
                # 通常: Yが大きい方(画面下側)がTip
                if by1 > by2: # by1の方が下にある
                    tip = (bx1, by1); tail = (bx2, by2)
                else: # by2の方が下にある
                    tip = (bx2, by2); tail = (bx1, by1)
                
                # 反転設定があれば逆にする
                if self.flip_tip:
                    tip, tail = tail, tip

                # 記録・ステータス
                status_color = (0, 255, 255)
                if self.is_recording:
                    self.angle_history.append(current_angle)
                    status_color = (0, 0, 255)
                    cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)

                if abs(current_angle - self.target_angle) < 5.0:
                    status_color = (0, 255, 0) # Good!

                # 針本体
                cv2.line(img, tail, tip, status_color, 6)
                
                # ガイド線（Tipから延長する）
                vec_x = tip[0] - tail[0]
                vec_y = tip[1] - tail[1]
                vec_len = np.sqrt(vec_x**2 + vec_y**2)
                
                if vec_len > 0:
                    unit_x = vec_x / vec_len
                    unit_y = vec_y / vec_len
                    pixel_len = self.guide_len_mm * self.PX_PER_MM
                    
                    guide_end_x = int(tip[0] + unit_x * pixel_len)
                    guide_end_y = int(tip[1] + unit_y * pixel_len)
                    
                    # ガイド線 (黄色い点線イメージの実線)
                    cv2.line(img, tip, (guide_end_x, guide_end_y), (255, 255, 0), 2)
                    cv2.circle(img, (guide_end_x, guide_end_y), 4, (255, 255, 0), -1)

                msg = f"{current_angle:.1f}"
                cv2.putText(img, msg, (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # 枠表示
            border_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), border_color, 2)
            self.last_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except:
            return frame

# --- メイン画面レイアウト ---
# カメラ映像
ctx = webrtc_streamer(
    key="needle-main",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=NeedleGuideSimulator,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

# --- ここに操作ボタンを集約（メインエリア） ---
st.markdown("### 🎮 操作パネル")

# Processorが動いている時だけ表示
if ctx.video_processor:
    ctx.video_processor.update_settings(roi_size, threshold, target_angle, flip_tip, guide_len_mm)

    # ボタンを横並びにする
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    # 1. テスト開始/終了ボタン
    if 'testing' not in st.session_state:
        st.session_state.testing = False

    with btn_col1:
        if not st.session_state.testing:
            if st.button("▶️ テスト開始", use_container_width=True, type="primary"):
                ctx.video_processor.start_test()
                st.session_state.testing = True
                st.rerun()
        else:
            if st.button("⏹️ 終了・採点", use_container_width=True, type="primary"):
                history = ctx.video_processor.stop_test()
                st.session_state.testing = False
                st.session_state.test_result = history
                st.rerun()

    # 2. 静止画保存ボタン
    with btn_col2:
        if st.button("📷 撮影", use_container_width=True):
            frame = ctx.video_processor.get_last_frame()
            if frame is not None:
                st.session_state.last_capture = frame
            else:
                st.toast("映像が見つかりません")

    # 3. リセットボタン
    with btn_col3:
        if st.button("🔄 リセット", use_container_width=True):
            if 'test_result' in st.session_state:
                del st.session_state.test_result
            if 'last_capture' in st.session_state:
                del st.session_state.last_capture
            st.rerun()

else:
    st.info("上の「START」を押してカメラを起動してください")

# --- 結果・画像表示エリア ---
st.markdown("---")

# キャプチャ画像の表示
if 'last_capture' in st.session_state:
    st.image(st.session_state.last_capture, caption="撮影画像", use_container_width=True)
    # ダウンロード用
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(st.session_state.last_capture, cv2.COLOR_RGB2BGR))
    if is_success:
        st.download_button("画像を保存", buffer.tobytes(), "puncture.png", "image/png")

# テスト結果の表示
if 'test_result' in st.session_state and st.session_state.test_result:
    data = st.session_state.test_result
    if len(data) > 5:
        df = pd.DataFrame(data, columns=["Angle"])
        avg = df["Angle"].mean()
        std = df["Angle"].std()
        score = max(0, int(100 - abs(avg - target_angle)*2 - std*5))
        
        st.success(f"🏆 スコア: {score} 点")
        cols = st.columns(2)
        cols[0].metric("平均角度", f"{avg:.1f}°", f"{avg - target_angle:.1f}")
        cols[1].metric("安定性(±)", f"{std:.2f}")
        st.line_chart(df)
    else:
        st.warning("データが短すぎます")
