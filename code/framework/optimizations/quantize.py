# ml_models/quantize.py
import tensorflow as tf

def quantize_model(saved_model_dir, output_dir):
    #Quantize a Tensorflow Model after training
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Speichert das quantisierte Modell
    with open(output_dir, 'wb') as f:
        f.write(tflite_model)
    print(f"Quantized model saved to {output_dir}")