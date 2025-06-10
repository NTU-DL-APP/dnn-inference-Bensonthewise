# train.py
import tensorflow as tf
import numpy as np
from pathlib import Path

# 1. 載入資料並正規化到 [0,1]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test .astype(np.float32) / 255.0

# 2. 定義純 MLP（不要 Dropout/BatchNorm/Conv）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512,  activation='relu'),
    tf.keras.layers.Dense(256,  activation='relu'),
    tf.keras.layers.Dense(128,  activation='relu'),
    tf.keras.layers.Dense(10,   activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. 訓練：30 epochs，batch_size=64
model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# 4. 評估
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}%")

# 5. 儲存成 HDF5
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
model.save(model_dir / "fashion_mnist.h5")
