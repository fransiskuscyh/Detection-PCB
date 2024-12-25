import tensorflow as tf

model = tf.keras.models.load_model('pcb_detection_cnn_modelv5.h5')  # Model CNN sebelumnya
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('pcb_detection_modelv5.tflite', 'wb') as f:
    f.write(tflite_model)
