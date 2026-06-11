# 💉 穿刺角度ガイドシミュレータ (Streamlit版)

透析穿刺の角度トレーニングを支援する Web アプリです。スマホ／PC のカメラ映像から
針の角度をリアルタイムに計測し、目標角度との一致度を採点します。

**公開URL:** https://senshi-app-8ccj2os5ilu6jmz2u3cfkw.streamlit.app/

---

## 主な機能

| 機能 | 説明 |
|------|------|
| 🤖 **AI針検出** | Teachable Machine で学習したモデル (ONNX) で「穿刺針あり／なし」を判定 |
| 🎯 **マーカー追従** | OpenCV CSRT トラッカーで針先・根本をロックオン追従し角度を計測 |
| 📐 **自動検出** | Hough 変換による直線検出で針の角度を自動計測 |
| 📊 **角度採点** | 記録した角度の平均・ばらつきからスコアを算出、時系列グラフ表示 |

## 構成ファイル

| ファイル | 役割 |
|----------|------|
| `app.py` | Streamlit アプリ本体 |
| `needle_model.onnx` | AI針検出モデル (約2MB)。`app.py` が起動時に読み込む |
| `requirements.txt` | Python 依存パッケージ |
| `runtime.txt` | Streamlit Cloud 用 Python バージョン指定 |
| `model_conversion/` | AIモデルの変換記録（再現用）。[詳細](model_conversion/README.md) |

> `nextjs-app/`（Vercel版の試作）と `ngrok.exe` はローカル参照用のため
> `.gitignore` で除外しています。

## ローカル起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## デプロイ (Streamlit Community Cloud)

`main` ブランチへの push で自動的に再デプロイされます。
カメラの NAT 越えが必要な環境向けに、無料の TURN サーバー
(Open Relay) をデフォルトで併用しています。安定運用したい場合は
Streamlit の **Secrets** に独自の TURN 設定を追加してください:

```toml
[turn]
urls = ["turn:your-turn-server:3478"]
username = "user"
credential = "pass"
```
