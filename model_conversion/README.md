# AIモデル変換記録

Teachable Machine で学習した画像分類モデルを、Streamlit (Python) から
利用できる **ONNX 形式** に変換するための作業フォルダです。

## モデル概要

- **学習元:** Google Teachable Machine (Image Project)
- **クラス:** `穿刺` (針あり) / `no 穿刺針` (針なし)
- **入力:** 224×224 RGB、前処理は `(pixel / 127.5) - 1.0`
- **出力:** 2クラスの確率 (softmax)

## ファイル

| ファイル | 内容 |
|----------|------|
| `model.json` | Teachable Machine の TFJS モデル構造 |
| `model.weights.bin` | 学習済み重み (バイナリ) |
| `metadata.json` | クラス名・入力サイズ等のメタdata |
| `convert.py` | TFJS → Keras → ONNX 変換スクリプト |

## 再変換の手順

```bash
python -m venv venv
venv/Scripts/python -m pip install "tensorflow-cpu==2.15.1" tf2onnx "onnx==1.16.1" "numpy<2"

# TFJS の重みを Keras モデルに復元して .h5 を出力
venv/Scripts/python convert.py

# .h5 を ONNX に変換 (※ ASCIIパス上で実行すること。
#   日本語パスだと TensorFlow のファイルIOが失敗する)
venv/Scripts/python -m tf2onnx.convert --keras needle_model.h5 \
    --output needle_model.onnx --opset 13
```

生成された `needle_model.onnx` をリポジトリ直下にコピーすると
`app.py` が読み込みます。

## モデルを更新したいとき

1. Teachable Machine で再学習し、`model.json` / `model.weights.bin` /
   `metadata.json` をエクスポートしてこのフォルダに上書き
2. 上記「再変換の手順」を実行
3. `needle_model.onnx` を直下へコピーして push
