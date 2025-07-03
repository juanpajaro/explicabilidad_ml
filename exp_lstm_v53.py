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

def recortar_hasta_punto(s_version):
    """
    Recorta el string recibido hasta el primer punto (no lo incluye).
    Ejemplo: "lstm_v224.h5" -> "lstm_v224"
    """
    return s_version.split('.')[0]

def crear_carpeta_si_no_existe(nombre_carpeta):
    """
    Recibe un string con el nombre de una carpeta.
    Si la carpeta no existe en la ruta actual, la crea.
    """
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
        print(f"Carpeta creada: {nombre_carpeta}")
    else:
        print(f"La carpeta ya existe: {nombre_carpeta}")
    return nombre_carpeta

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


def sumar_por_valor(arr1, arr2, absoluto=False):
    """
    #Suma los valores de arr2 agrupados por el valor y posición correspondiente en arr1.
    #Retorna una lista de diccionarios con el valor de arr1 como clave y la suma de arr2 como valor.
    #Imprime k y v a medida que se agregan.
    #Ejemplo de retorno: [{k: suma}, ...]
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Los arreglos deben tener el mismo tamaño.")    
    for k, v in zip(arr1, arr2):
        encontrado = False
        v_abs = abs(v) if absoluto else v
        for d in lista_resultado:
            if k in d:
                d[k] += v_abs
                #print(f"Actualizando: {k} -> {d[k]}")
                encontrado = True
                break
        if not encontrado:
            lista_resultado.append({k: v_abs})
            #print(f"Agregando: {k} -> {v}")
    return lista_resultado


def importancia_acumulada(lista_diccionarios, nuevos_diccionarios):
    """
    Acumula la importancia de los valores de una lista de nuevos diccionarios en lista_diccionarios.
    Si la clave ya existe, suma el valor; si no, la agrega.
    Retorna la lista actualizada de diccionarios únicos.
    """
    # Convertir lista de diccionarios acumulados a un solo diccionario
    acumulado = {}
    for d in lista_diccionarios:
        for k, v in d.items():
            if k in acumulado:
                acumulado[k] += v
            else:
                acumulado[k] = v
    # Iterar sobre la lista de nuevos diccionarios y acumular
    for d in nuevos_diccionarios:
        for k, v in d.items():
            if k in acumulado:
                acumulado[k] += v
            else:
                acumulado[k] = v
    # Convertir de nuevo a lista de diccionarios únicos
    lista_resultado = [{k: acumulado[k]} for k in acumulado]
    return lista_resultado

def pintar_grafica_por_paciente(X_seq, token_imp, version, predictions, label, i, n_carpeta):
    """
    Dibuja una gráfica de barras horizontales para las importancias de tokens de un paciente.
    Muestra solo el valor sumado (por token único) al final de cada barra y guarda la imagen como archivo PNG.
    """
    
    # Agrupar importancias por token único
    from collections import defaultdict
    suma_por_token = defaultdict(float)
    for token, imp in zip(X_seq, token_imp):
        suma_por_token[token] += imp

    tokens_unicos = list(suma_por_token.keys())
    importancias_sumadas = [suma_por_token[token] for token in tokens_unicos]

    print("pintando la grafica")
    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(tokens_unicos, importancias_sumadas)
    ax.set_xlabel('Importancia')
    ax.set_title(f'Atribuciones de Integrated Gradients\nModel: {version} | Pred: {predictions} | label: {label}')

    # Añadir solo el valor sumado al final de cada barra
    for bar, suma in zip(bars, importancias_sumadas):
        width = bar.get_width()
        ax.annotate(f'{suma:.2f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va='center',
                    fontsize=8)

    nombre_grafica = f"grafica_{version}_{i}.png"
    print("nombre grafica:", nombre_grafica)
    plt.savefig(n_carpeta+"/"+nombre_grafica, dpi=300, bbox_inches='tight')
    plt.close(fig)

def graficar_lista_diccionarios(lista_diccionarios, titulo="Importancia acumulada", nombre_grafica="grafica_acumulada.png", vocabulary=None, top_n=None, save_file=None):
    """
    Grafica una lista de diccionarios donde los keys son las etiquetas (x) y los values son los valores (y).
    Los datos se muestran de menor a mayor y se puede limitar el número de elementos mostrados con top_n.
    """

    # Unir todos los diccionarios en uno solo para evitar duplicados
    datos = {}
    for d in lista_diccionarios:
        for k, v in d.items():
            datos[k] = v

    # Ordenar los datos de menor a mayor valor
    datos_ordenados = sorted(datos.items(), key=lambda item: item[1])

    # Si se especifica top_n, tomar solo los últimos top_n elementos (los de mayor valor)
    if top_n is not None:
        datos_ordenados = datos_ordenados[-top_n:]

    etiquetas = [k for k, v in datos_ordenados]
    valores = [v for k, v in datos_ordenados]

    # Convertir etiquetas si hay vocabulario
    if vocabulary is not None:
        etiquetas_n = convert_to_text(np.array(etiquetas).reshape(1, -1), vocabulary)
    else:
        etiquetas_n = etiquetas

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(etiquetas_n, valores)
    ax.set_xlabel('Importancia')
    ax.set_title(titulo)

    # Añadir el valor al final de cada barra
    for bar, valor in zip(bars, valores):
        width = bar.get_width()
        ax.annotate(f'{valor:.2f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va='center',
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(save_file+"/"+nombre_grafica, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada como {nombre_grafica}")

if __name__ == "__main__":
    
    version = "lstm_v977.h5"
    s_version = recortar_hasta_punto(version)
    print("s_version:", s_version)
    n_carpeta = crear_carpeta_si_no_existe("./g_train"+"/"+ s_version)
    print("name_carpeta:", n_carpeta)

    ruta_modelo = "./models/"+version
    ruta_performance = "/home/pajaro/compu_Pipe_V3/performance_zine/performance_report_osa.csv"    
    

    folder = buscar_valor_performance(ruta_csv=ruta_performance, columna_busqueda="model_name", valor_busqueda=version, columna_retorno="path_vectorization")
    print("folder:", folder)

    folder_recortado = recortar_folder(folder, "compu_Pipe_V3", incluir_carpeta=False)
    print("folder_recortado:", folder_recortado)

    folder_n = "./"+folder_recortado+"/"
    print("folder_n:", folder_n)

    vocabulary = load_numpy_array(folder_n + "vocab.npy")
    print("vocabulary size:", len(vocabulary))
    X_train = load_numpy_array(folder_n + "X_train.npy")
    print("X_train shape:", X_train.shape)
    X_test = load_numpy_array(folder_n + "X_test.npy")
    print("X_test shape:", X_test.shape)
    y_train = load_numpy_array(folder_n + "y_train.npy")
    y_test = load_numpy_array(folder_n + "y_test.npy")
    print("y_test", y_test)
    print("y_train", y_train)
    print("posiciones_pos", posiciones_de_uno(y_train))
    print("len pos", len(posiciones_de_uno(y_train)))

    
    model = tf.keras.models.load_model(ruta_modelo)
    model.summary()

    #print(X_train[:10, :])

    #l_patient = [145, 178, 171, 184, 179, 187]
    #l_patient = [145, 178]
    #i = 187
    X_analysis = X_train
    l_patient = np.arange(len(X_analysis)).tolist()
    lista_resultado = []

    for i in l_patient:
        print("i:", i)
        X_sample = X_analysis[i:i+1, :]
        print("X_sample shape:", X_sample.shape)    
        predictions = model.predict(X_sample)
        print("predictions shape:", predictions.shape)
        print(predictions)
        predictions = (predictions >= 0.5).astype(int)
        predictions = predictions.item()
        #predictions = predictions[1]
        #print("pred:", predictions)
        #print("pred shape:", predictions.shape)
        #print("target:", y_train[i:i+1])
        label = y_train[i:i+1]

        t_pred = model.predict(X_train)
        #t_pred = t_pred.argmax(axis=1)
        t_pred = (t_pred >= 0.5).astype(int)
        #print("t_pred:", t_pred)
        print("t_pred shape:", t_pred.shape)
        #print("t_pred count unique:", np.unique(t_pred, return_counts=True))

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
        print('Token_imp shape:', token_imp.shape)
        #attrs = attributions.sum(axis=2)
        #print('Attrs shape:', attrs.shape)    
        
        print("X_sample", X_sample)
        X_seq = convert_to_text(X_sample, vocabulary)
        print("X_seq:", X_seq)
        #print("X_seq type:", type(X_seq))
        print("X_seq shape:", len(X_seq))
        #print("X_seq set:", set(X_seq))
        print("X_seq_set len:", len(set(X_seq)))    
        #attrs = token_imp[i]

        X_sample_f = X_sample.flatten()
        print("X_sample_f:", X_sample_f.shape)
        lista_resultado = sumar_por_valor(X_sample_f, token_imp, absoluto=False)
        print("l_phe_patient:", lista_resultado)        


        """
        print("pintando la grafica")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(X_seq, token_imp)
        ax.set_xlabel('Importancia')
        ax.set_title(f'Atribuciones de Integrated Gradients\nModel: {version} | Pred: {predictions} | label: {label}')
        print("nombre grafica:", "grafica_"+version+"_"+str(i)+".png")
        plt.savefig("grafica_"+version+"_"+str(i)+".png", dpi=300, bbox_inches='tight')
        #plt.show()
        """

        pintar_grafica_por_paciente(X_seq, token_imp, version, predictions, label, i, n_carpeta)

    graficar_lista_diccionarios(lista_resultado, titulo="Global attribution concepts train", nombre_grafica="grafica_fenotipos_importantes_train_sin_abs_v1.png", vocabulary=vocabulary, top_n=20, save_file=n_carpeta)


    


