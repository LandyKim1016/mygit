import tensorflow as tf

# GPU 사용 가능 여부 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU를 사용할 수 있습니다: {gpus}")
else:
    print("GPU를 사용할 수 없습니다.")
