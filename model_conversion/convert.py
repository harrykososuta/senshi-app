# Teachable Machine (TFJS layers形式) -> Keras h5 -> ONNX 変換スクリプト
import json
import numpy as np
import tensorflow as tf

DTYPE_SIZE = {"float32": (np.float32, 4), "int32": (np.int32, 4), "uint8": (np.uint8, 1)}

with open("model.json", "r", encoding="utf-8") as f:
    mj = json.load(f)

topology = mj["modelTopology"]
manifest = mj["weightsManifest"]

# Keras モデルを topology から再構築
model = tf.keras.models.model_from_json(json.dumps(topology))
model.build((None, 224, 224, 3))
print("Model rebuilt:", model.input_shape, "->", model.output_shape)

# weights バイナリを読み込み、manifest 順に切り出す
raw = open("model.weights.bin", "rb").read()
weights_by_name = {}
offset = 0
for group in manifest:
    for spec in group["weights"]:
        np_dtype, size = DTYPE_SIZE[spec["dtype"]]
        count = int(np.prod(spec["shape"])) if spec["shape"] else 1
        arr = np.frombuffer(raw, dtype=np_dtype, count=count, offset=offset)
        arr = arr.reshape(spec["shape"])
        weights_by_name[spec["name"]] = arr
        offset += count * size
print(f"Parsed {len(weights_by_name)} weight tensors, {offset} bytes (file: {len(raw)})")
assert offset == len(raw), "weight binary size mismatch"

# 名前で対応付けて代入
missing = []
assigned = 0
for w in model.weights:
    key = w.name.split(":")[0]
    if key in weights_by_name:
        w.assign(weights_by_name[key])
        assigned += 1
    else:
        missing.append(key)
print(f"Assigned {assigned}/{len(model.weights)} weights")
if missing:
    print("MISSING:", missing[:10])
    raise SystemExit(1)

# 動作確認 (ダミー入力)
dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
pred = model.predict(dummy, verbose=0)
print("Sanity check prediction:", pred, "sum:", pred.sum())

model.save("needle_model.h5")
print("Saved needle_model.h5")
