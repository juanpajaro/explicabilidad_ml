#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import os
import pandas as pd

def load_numpy_array(file_path):
    """
    Load a numpy array from a .npy file.
    """
    return np.load(file_path)

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