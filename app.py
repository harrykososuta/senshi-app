import streamlit as st
import cv2
import numpy as np
import av
import time
import pandas as pd # グラフ描画用
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- UI設定 ---
st.set_page_config(page_title="穿刺ガイド - スコアリングモード", layout="wide")
st.title("💉 穿刺ガイド - スコアリング＆記録")
st.caption("Ver 5.0 - Test & Score Mode")

# --- 通信設定 (Metered.ca または Google) ---
# ※ここに前回の Metered.ca の設定を入れてください
TURN_USERNAME = "【ここにusername】"
TURN_PASSWORD = "【ここにpassword】"

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # Meteredの設定があればコメントアウトを外して使う
            # {
            #     "urls": ["turn:global.turn.metered.ca:80", "turn:global.turn.metered.ca:443"],
            #     "username": TURN_USERNAME,
            #     "credential": TURN_PASSWORD,
            # },
        ]
    }
)

# --- サイドバー設定 ---
st.sidebar.header("⚙️ 設定")

# 1. 認識設定
st.sidebar.subheader("🎥 認識・カメラ")
roi_size = st.sidebar.slider("検出枠サイズ (%)", 10, 100, 40)
threshold = st.sidebar.slider("検出感度", 30, 150, 50)
camera_mode = st.sidebar.radio("カメラ向き", ("自分側", "外側"), index=1)
if camera_mode == "自分側":
    video_constraints = {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}}
else:
    video_constraints = {"facingMode": "environment", "width": {"ideal": 640}, "height": {"ideal": 480}}

# 2. テスト設定
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 テスト基準")
target_angle = st.sidebar.number_input("目標角度 (度)", 20.0, 50.0, 30.0, step=1.0)
st.sidebar.caption(f"目標: {target_angle}度 をキープしてください")

# --- 映像処理クラス ---
class NeedleGuideSimulator(VideoProcessorBase):
    def __init__(self):
        # 設定値
        self.roi_percent = 40
        self.threshold = 50
        self.target_angle = 30.0
        
        # 状態管理
        self.is_recording = False
        self.angle_history = [] # テスト中の角度データ
        self.last_frame = None  # 静止画保存用

    def update_settings(self, roi, thresh, target):
        self.roi_percent = roi
        self.threshold = thresh
        self.target_angle = target

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

            # マスク処理
            mask = np.zeros_like(img)
            cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), -1)
            masked_img = cv2.bitwise_and(img, mask)

            # 画像処理
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

            # --- 描画とデータ記録 ---
            status_color = (0, 255, 255) # 黄色（通常）

            if current_angle is not None:
                # テスト中ならデータを記録
                if self.is_recording:
                    self.angle_history.append(current_angle)
                    status_color = (0, 0, 255) # 赤色（録画中）
                    cv2.circle(img, (30, 30), 15, (0, 0, 255), -1) # RECマーク

                # ターゲット角度に近いと緑色にする
                if abs(current_angle - self.target_angle) < 5.0:
                    status_color = (0, 255, 0)

                # 描画
                bx1, by1, bx2, by2 = best_line[0]
                if by1 < by2: tip = (bx1, by1); tail = (bx2, by2)
                else: tip = (bx2, by2); tail = (bx1, by1)
                
                cv2.line(img, tail, tip, status_color, 6)
                msg = f"Angle: {current_angle:.1f}"
                cv2.putText(img, msg, (tail[0], tail[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # ROI枠表示
            border_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), border_color, 2)
            
            # ターゲット角度表示
            cv2.putText(img, f"Target: {self.target_angle}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # 静止画保存用に現在のフレームを保持（BGR->RGB）
            self.last_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            return frame

# --- メイン画面構成 ---

col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="needle-test-mode",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=NeedleGuideSimulator,
        media_stream_constraints={"video": video_constraints, "audio": False},
        async_processing=True,
    )

# --- 操作パネル（右カラム） ---
with col2:
    st.subheader("📸 記録 & テスト")
    
    # Processorが起動しているか確認
    if ctx.video_processor:
        # 設定をリアルタイム反映
        ctx.video_processor.update_settings(roi_size, threshold, target_angle)

        # --- A. 静止画保存機能 ---
        if st.button("📷 今の画面を保存"):
            frame = ctx.video_processor.get_last_frame()
            if frame is not None:
                # 画像を表示してダウンロードボタンを出す
                st.image(frame, channels="RGB", use_container_width=True)
                # 画像をバイト列に変換してダウンロード可能にする
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if is_success:
                    st.download_button(
                        label="画像をダウンロード",
                        data=buffer.tobytes(),
                        file_name="puncture_shot.png",
                        mime="image/png"
                    )
            else:
                st.warning("映像が見つかりません")

        st.markdown("---")

        # --- B. テスト機能 ---
        # セッション状態でテスト中かどうか管理
        if 'testing' not in st.session_state:
            st.session_state.testing = False

        if not st.session_state.testing:
            if st.button("▶️ テスト開始", type="primary"):
                ctx.video_processor.start_test()
                st.session_state.testing = True
                st.rerun()
        else:
            st.warning("🔴 測定中... 角度をキープしてください")
            if st.button("⏹️ テスト終了"):
                history = ctx.video_processor.stop_test()
                st.session_state.testing = False
                st.session_state.test_result = history # 結果を保存
                st.rerun()

    else:
        st.info("カメラを開始してください")

# --- 結果表示エリア（テスト終了後） ---
if 'test_result' in st.session_state and st.session_state.test_result:
    data = st.session_state.test_result
    st.markdown("---")
    st.header("📊 テスト結果")

    if len(data) < 5:
        st.error("データ不足です。もう少し長く測定してください。")
    else:
        # データ分析
        df = pd.DataFrame(data, columns=["Angle"])
        
        # 指標計算
        avg_angle = df["Angle"].mean()
        std_dev = df["Angle"].std() # 標準偏差（ブレの大きさ）
        
        # スコア計算（簡易ロジック）
        # 1. 正確性: 目標とのズレ 1度につき 5点減点
        accuracy_score = max(0, 50 - abs(avg_angle - target_angle) * 5)
        
        # 2. 安定性: ブレ(標準偏差) 1.0につき 10点減点
        stability_score = max(0, 50 - std_dev * 10)
        
        total_score = int(accuracy_score + stability_score)

        # 結果表示
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("総合スコア", f"{total_score} / 100")
        res_col2.metric("平均角度", f"{avg_angle:.1f}°", delta=f"{avg_angle - target_angle:.1f}")
        res_col3.metric("安定性(ブレ)", f"±{std_dev:.2f}°", help="値が小さいほど手が安定しています")

        # グラフ
        st.line_chart(df)
        
        # コメント
        if total_score >= 80:
            st.balloons()
            st.success("素晴らしい！プロ級の穿刺技術です。")
        elif total_score >= 60:
            st.info("良好です。もう少しブレを抑えられると完璧です。")
        else:
            st.warning("ブレが大きいです。脇を締めて固定してみましょう。")
