#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt
import pandas as pd
import os

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
    ig = IntegratedGradients(model, layer=embedding_layer, n_steps=50, method='gausslegendre', internal_batch_size=100)
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


def buscar_valor_performance(ruta_csv, columna_busqueda, valor_busqueda, columna_retorno):
    """
    Busca un valor específico en una columna de un archivo CSV y retorna el valor de otra columna de la misma fila.

    Parámetros:
    ruta_csv (str): Ruta del archivo CSV.
    columna_busqueda (str): Nombre de la columna donde se buscará el valor.
    valor_busqueda (str): Valor que se desea buscar.
    columna_retorno (str): Nombre de la columna cuyo valor se desea obtener.

    Retorna:
    El valor encontrado en la columna_retorno o None si no se encuentra.
    """
    try:
        df = pd.read_csv(ruta_csv)
        print("df.columns:", df.columns)
        print("df.shape:", df.shape)
        fila = df[df[columna_busqueda] == valor_busqueda]
        if not fila.empty:
            return fila.iloc[0][columna_retorno]
        else:
            print("Valor no encontrado.")
            return None
    except FileNotFoundError:
        print("Archivo no encontrado.")
    except KeyError as e:
        print(f"Columna no encontrada: {e}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
    return None

def recortar_folder(ruta_completa, nombre_carpeta, incluir_carpeta=True):
    """
    Recorta una ruta a partir del nombre de una carpeta.

    Parámetros:
    ruta_completa (str): Ruta completa del archivo o directorio.
    nombre_carpeta (str): Nombre de la carpeta que se quiere identificar.
    incluir_carpeta (bool): Si True, incluye la carpeta en el resultado; si False, la excluye.

    Retorna:
    str: Ruta recortada o None si la carpeta no se encuentra.
    """
    partes = ruta_completa.split(os.sep)
    try:
        indice = partes.index(nombre_carpeta)
        if incluir_carpeta:
            return os.sep.join(partes[:indice + 1])
        else:
            return os.sep.join(partes[indice + 1:])
    except ValueError:
        print("Carpeta no encontrada en la ruta.")
        return None
    
def posiciones_de_uno(arr):
    """Devuelve las posiciones (índices) donde el valor es 1 en un array NumPy."""
    return np.where(arr == 1)[0]

if __name__ == "__main__":
    
    version = "lstm_v224.h5"
    #145, 178, 171, 184
    i = 187
    ruta_modelo = "./models/"+version
    ruta_performance = "/home/pajaro/compu_Pipe_V3/performance_zine/performance_report.csv"    
    

    folder = buscar_valor_performance(ruta_csv=ruta_performance, columna_busqueda="model_name", valor_busqueda=version, columna_retorno="path_vectorization")
    print("folder:", folder)

    folder_recortado = recortar_folder(folder, "compu_Pipe_V3", incluir_carpeta=False)
    print("folder_recortado:", folder_recortado)

    folder_n = "./"+folder_recortado+"/"
    print("folder_n:", folder_n)

    vocabulary = load_numpy_array(folder_n + "vocab.npy")
    X_train = load_numpy_array(folder_n + "X_train.npy")
    X_test = load_numpy_array(folder_n + "X_test.npy")
    y_train = load_numpy_array(folder_n + "y_train.npy")
    y_test = load_numpy_array(folder_n + "y_test.npy")
    print("y_test", y_test)
    print("y_train", y_train)
    print("posiciones_pos", posiciones_de_uno(y_train))
    print("len pos", len(posiciones_de_uno(y_train)))

    
    model = tf.keras.models.load_model(ruta_modelo)
    model.summary()

    #print(X_train[:10, :])
    
    X_sample = X_train[i:i+1, :]
    print("X_sample shape:", X_sample.shape)    
    predictions = model.predict(X_sample)
    print("predictions shape:", predictions.shape)
    print(predictions)
    predictions = (predictions >= 0.5).astype(int)
    predictions = predictions.item()
    #predictions = predictions[1]
    print("pred:", predictions)
    #print("pred shape:", predictions.shape)
    print("target:", y_train[i:i+1])
    label = y_train[i:i+1]

    t_pred = model.predict(X_train)
    #t_pred = t_pred.argmax(axis=1)
    t_pred = (t_pred >= 0.5).astype(int)
    #print("t_pred:", t_pred)
    print("t_pred shape:", t_pred.shape)
    print("t_pred count unique:", np.unique(t_pred, return_counts=True))

    # 6. Ejecutar IG
    embedding_layer = model.layers[0]
    #embedding_layer = model.get_layer('embedding_1')
    print("cargue de Integrated Gradients")
    ig = load_model_ig(model, embedding_layer)
    # 7. Explicación
    base = np.zeros_like(X_sample, dtype=np.float32)
    print("calculando la explicacion")    
    explanation = ig.explain(X_sample, baselines=0, target=predictions)

    # 7. Visualizar las atribuciones
    print("calculando las atribuciones")
    attributions = explanation.attributions[0]  # (100, embedding_dim)
    print('Attributions shape:', attributions.shape)
    token_importances = np.sum(attributions, axis=-1)  # suma por embedding
    print('Token importances shape:', token_importances.shape)
    # Asegurarse de que token_importances es un array 1D
    token_imp = np.array(token_importances).flatten()
    #attrs = attributions.sum(axis=2)
    #print('Attrs shape:', attrs.shape)    

    
    print("X_sample", X_sample)
    X_seq = convert_to_text(X_sample, vocabulary)
    print("X_seq:", X_seq)
    print("X_seq shape:", len(X_seq))
    print("X_seq set:", set(X_seq))
    #attrs = token_imp[i]


    print("pintando la grafica")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(X_seq, token_imp)
    ax.set_xlabel('Importancia')
    ax.set_title(f'Atribuciones de Integrated Gradients\nModel: {version} | Pred: {predictions} | label: {label}')
    print("nombre grafica:", "grafica_"+version+"_"+str(i)+".png")
    plt.savefig("grafica_"+version+"_"+str(i)+".png", dpi=300, bbox_inches='tight')
    #plt.show()

