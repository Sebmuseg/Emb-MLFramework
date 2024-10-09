# conversion/tf_to_onnx.py
import tf2onnx

def convert_tf_model_to_onnx(saved_model_dir, output_path):
    """Konvertiert ein TensorFlow-Modell in ONNX-Format."""
    model_proto, _ = tf2onnx.convert.from_saved_model(saved_model_dir, output_path=output_path)
    print(f"TensorFlow model converted to ONNX and saved to {output_path}")