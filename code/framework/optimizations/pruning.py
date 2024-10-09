# ml_models/pruning.py
import tensorflow_model_optimization as tfmot
import tensorflow as tf

def apply_pruning(model):
    #Executes Pruting on a available model
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Pruning-Konfiguration
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30, final_sparsity=0.80, begin_step=0, end_step=1000)
    }

    # Anwendunung von Pruning auf das Modell
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
    
    return model_for_pruning