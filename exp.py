#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt


def load_numpy_array(file_path):
    """
    Load a numpy array from a .npy file.
    """
    return np.load(file_path)

def load_model_lstm(file_path):
    """
    Load a TensorFlow model from a .h5 file.
    """
    return tf.keras.models.load_model(file_path)

def load_model_ig(model, embedding_layer):
    ig = IntegratedGradients(model, layer=embedding_layer, n_steps=50, method='gausslegendre', internal_batch_size=1)
    return ig

def convert_to_text(X_train, vocabulary):
    X_seq = []
    patient_n = 0
    for i in range(len(X_train[patient_n])):
            number = X_train[patient_n][i]
            #print('number:', number)
            enf = vocabulary[number]
            X_seq.append(enf)
    return X_seq

if __name__ == "__main__":
     
    vocabulary = load_numpy_array("./tokens/vocab.npy")
    X_train = load_numpy_array("./tokens/X_train.npy")
    X_test = load_numpy_array("./tokens/X_test.npy")
    y_train = load_numpy_array("./tokens/y_train.npy")
    y_test = load_numpy_array("./tokens/y_test.npy")

    ruta_modelo = './lstm_v66.h5'
    model = tf.keras.models.load_model(ruta_modelo)
    model.summary()

    X_sample = X_train[:1, :]
    predictions = model(X_sample).numpy().argmax(axis=1)
    print(predictions)

    # 6. Ejecutar IG
    embedding_layer = model.get_layer('embedding_1')
    ig = load_model_ig(model, embedding_layer)
    # 7. Explicación
    explanation = ig.explain(X_sample, target=predictions)

    # 7. Visualizar las atribuciones
    attributions = explanation.attributions[0]  # (100, embedding_dim)
    token_importances = np.sum(attributions, axis=-1)  # suma por embedding

    X_seq = convert_to_text(X_sample, vocabulary)
    # Asegurarse de que token_importances es un array 1D
    token_imp = np.array(token_importances).flatten()

    """"

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(X_seq, token_imp)
    ax.set_xlabel('Importancia')
    ax.set_title('Atribuciones de Integrated Gradients')
    plt.savefig('grafica.png', dpi=300, bbox_inches='tight')
    #plt.show()
    """
