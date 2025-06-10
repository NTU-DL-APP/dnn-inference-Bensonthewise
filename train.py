# train.py
import tensorflow as tf
import numpy as np
from pathlib import Path

# 1. 載入資料，並增加 channel 維度
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test .astype(np.float32) / 255.0
# CNN 需要 (H,W,1) 的輸入格式
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test,  axis=-1)

# 2. 定義 CNN 模型
model = tf.keras.Sequential([
    # 卷積 + 池化 block 1
    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same',
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=2),
    # 卷積 + 池化 block 2
    tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    # 展平 + 全連接
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. 訓練：延長到 15–20 epochs 即可
model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# 4. 評估
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}%")

# 5. 匯出
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
model.save(model_dir / "fashion_mnist.h5")
