# ml_models/tflite_model.py
import tensorflow as tf
from pathlib import Path
from utils.logging_utils import log_deployment_event
import numpy as np

class TFLiteModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the TensorFlow Lite model. Load the model from a file if a model_path is provided,
        otherwise use the provided TFLite interpreter instance.

        Parameters:
        - model_path: Path to the TFLite model file (optional).
        - model: An existing TensorFlow Lite Interpreter instance (optional).
        """
        if model_path:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)  # Load the TFLite model
            self.interpreter.allocate_tensors()  # Allocate memory for model tensors
        elif model:
            self.interpreter = model  # Use the provided TensorFlow Lite interpreter instance
        else:
            raise ValueError("Either model_path or model must be provided.")
        
        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def predict(self, input_data):
        """
        Predict using the TensorFlow Lite model.

        Parameters:
        - input_data: Input data to be passed to the TFLite model.

        Returns:
        - The model's prediction.
        """
        # Get input and output details for the interpreter
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Set the input tensor
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        # Invoke the interpreter (run the model)
        self.interpreter.invoke()

        # Get the output tensor
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def save(self, file_name):
        """
        Save the TensorFlow Lite model to disk.

        Parameters:
        - file_name: The file name to save the TFLite model.

        Returns:
        - A dictionary with the status and the path of the saved model.
        """
        # Create the complete file path with a `.tflite` extension
        file_path = self.data_dir / file_name.with_suffix('.tflite')
        
        try:
            # Open the file in binary mode and save the TFLite model (FlatBuffers format)
            with open(file_path, 'wb') as f:
                f.write(self.interpreter._get_model())  # This assumes `self.interpreter` has the model loaded
            
            # Log the deployment event after a successful save
            log_deployment_event(f"TFLite model saved to {file_path}")
            
            # Return a success status with the saved model path
            return {"status": "success", "model_path": str(file_path)}

        except Exception as e:
            # Log any errors that occurred during the saving process
            log_deployment_event(f"Error saving TFLite model: {str(e)}", log_level="error")
            
            # Return an error status with the error message
            return {"status": "error", "message": str(e)}
        
    def train(self, train_data, train_labels, validation_data=None, validation_labels=None, epochs=10, batch_size=32, model_save_name="trained_model.tflite"):
        """
        Train a TensorFlow model and convert it to TensorFlow Lite.

        Parameters:
        - train_data: Training data (numpy array or tf.data.Dataset).
        - train_labels: Labels corresponding to the training data.
        - validation_data: Validation data (optional).
        - validation_labels: Labels corresponding to the validation data (optional).
        - epochs: Number of epochs to train the model (default: 10).
        - batch_size: Batch size for training (default: 32).
        - model_save_name: The name to save the converted TensorFlow Lite model as.

        Returns:
        - A dictionary with the status and the path of the saved TensorFlow Lite model.
        """
        try:
            # Define a basic TensorFlow model (or load a pre-defined one)
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=train_data.shape[1:]),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(set(train_labels)), activation='softmax')
            ])
            
            # Compile the TensorFlow model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the TensorFlow model
            model.fit(
                train_data, 
                train_labels, 
                validation_data=(validation_data, validation_labels) if validation_data is not None else None, 
                epochs=epochs, 
                batch_size=batch_size
            )
            
            # Convert the trained model to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            # Save the TensorFlow Lite model to disk
            file_path = self.data_dir / model_save_name
            with open(file_path, 'wb') as f:
                f.write(tflite_model)

            # Log the deployment event
            log_deployment_event(f"TFLite model saved to {file_path}")
            
            return {"status": "success", "model_path": str(file_path)}

        except Exception as e:
            # Log any errors that occur during the training or conversion process
            log_deployment_event(f"Error during TFLite model training: {str(e)}", log_level="error")
            
            # Return an error status with the exception message
            return {"status": "error", "message": str(e)}
        
    def evaluate(self, eval_data, eval_labels):
        """
        Evaluate the TensorFlow Lite model using the provided data and labels.

        Parameters:
        - eval_data: Features for evaluation (NumPy array).
        - eval_labels: True labels for evaluation (NumPy array).
        
        Returns:
        - A dictionary with evaluation metrics (e.g., accuracy).
        """
        try:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            correct_predictions = 0

            for i in range(len(eval_data)):
                # Set input tensor
                self.interpreter.set_tensor(input_details[0]['index'], [eval_data[i]])
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output tensor
                output = self.interpreter.get_tensor(output_details[0]['index'])
                predicted_label = np.argmax(output)

                # Compare predicted label to the actual label
                if predicted_label == eval_labels[i]:
                    correct_predictions += 1

            accuracy = correct_predictions / len(eval_labels)
            return {"accuracy": accuracy}

        except Exception as e:
            return {"status": "error", "message": str(e)}