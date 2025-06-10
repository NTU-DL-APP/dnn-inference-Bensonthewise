# export.py
import json, numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

model_dir = Path("model")
model    = load_model(model_dir/"fashion_mnist.h5")

layer_list = []
weight_dict = {}

for layer in model.layers:
    cls_name = layer.__class__.__name__       # Flatten 或 Dense
    name     = layer.name                    # e.g. "flatten", "dense"
    cfg       = {}
    wnames    = []

    if cls_name == "Dense":
        # 取 activation 名稱
        act = layer.activation.__name__      # "relu" 或 "softmax"
        cfg["activation"] = act

        # 取出該層的權重與偏差
        W, b = layer.get_weights()
        wkey, bkey = f"{name}_W", f"{name}_b"
        weight_dict[wkey] = W
        weight_dict[bkey] = b
        wnames = [wkey, bkey]

    # Flatten 層不需要 activation，也沒權重
    layer_list.append({
        "name":   name,
        "type":   cls_name,
        "config": cfg,
        "weights": wnames
    })

# 寫出 JSON
with open(model_dir/"fashion_mnist.json", "w") as f:
    json.dump(layer_list, f, indent=2)

# 寫出 .npz 權重
np.savez(model_dir/"fashion_mnist.npz", **weight_dict)

print("✅ Export 完成，請確認 model/ 下有：")
print("   • fashion_mnist.json  （layers list）")
print("   • fashion_mnist.npz   （權重檔）")
