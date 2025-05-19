#!/usr/bin/env python3
import utils
import numpy as np


def convert_to_words(sequence: np.ndarray, vocabulary: np.ndarray) -> list:
    """
    Convierte una secuencia de enteros a palabras usando un vocabulario (np.array).

    Parámetros:
    - sequence: np.ndarray → Secuencia de enteros.
    - vocabulary: np.ndarray → Array de palabras (índice = entero).

    Retorna:
    - List[str]: Palabras correspondientes.
    """
    return [vocabulary[token] if token < len(vocabulary) else "<UNK>" for token in sequence]


def find_matching_sequences(
    sequences: list,
    terms: list,
    vocabulary: np.ndarray,
    labels: np.ndarray
) -> list:
    """
    Busca coincidencias parciales en cada secuencia y retorna resultados detallados.

    Parámetros:
    - sequences: List[np.ndarray] → Lista de secuencias de enteros.
    - terms: List[str] → Lista de términos a buscar (coincidencia parcial).
    - vocabulary: np.ndarray → Vocabulario que convierte enteros a palabras.
    - labels: np.ndarray → Array de etiquetas correspondientes a cada secuencia.

    Retorna:
    - List[Tuple[int, Any, int, List[str]]]: (índice, etiqueta, número de coincidencias, palabras coincidentes)
    """
    results = []

    for idx, (seq, label) in enumerate(zip(sequences, labels)):
        words = convert_to_words(seq, vocabulary)
        matched_words = [word for word in words if any(term in word for term in terms)]
        match_count = len(matched_words)

        if match_count > 0:
            #print(f"Secuencia {idx} (Label: {label}): {words}")
            #print(f"  → Coincidencias: {match_count}")
            #print(f"  → Términos que coinciden: {matched_words}\n")
            results.append((idx, label, match_count, matched_words))

    return results

def main():
    # Cargar vocabulario y secuencias
    version = "lstm_v165.h5"
    #ruta_performance = "/home/pajaro/compu_Pipe_V3/performance_zine/performance_report.csv"
    ruta_performance = "/home/pajaro/compu_Pipe_V3/performance_zine/performance_report.csv"

    folder = utils.buscar_valor_performance(ruta_csv=ruta_performance, columna_busqueda="model_name", valor_busqueda=version, columna_retorno="path_vectorization")
    print("folder:", folder)

    folder_recortado = utils.recortar_folder(folder, "compu_Pipe_V3", incluir_carpeta=False)
    print("folder_recortado:", folder_recortado)

    folder_n = "./"+folder_recortado+"/"
    print("folder_n:", folder_n)

    vocabulary = utils.load_numpy_array(folder_n + "vocab.npy")
    print("vocabulary size:", len(vocabulary))
    X_train = utils.load_numpy_array(folder_n + "X_train.npy")
    print("X_train shape:", X_train.shape)
    #print("X_train:", X_train[:10, :])
    #X_samples = X_train[:3, :]
    #print("X_samples", X_samples)
    y_train = utils.load_numpy_array(folder_n + "y_train.npy")
    print("y_train shape:", y_train.shape)
    print("y_train count unique:", np.unique(y_train, return_counts=True))
    labels = y_train


    """
    for idx, seq in enumerate(X_samples):
        words = convert_to_words(seq, vocabulary)
        print(f"Sec {idx}: {words}")

    """
    
    # Definir términos a buscar
    #terms = ["diabetes", "hipertension", "obesidad", "tumor", "laringitis"]
    #terms = ["diabetes", "hipertension", "obesidad"]
    #terms = ["hipersomnia","sindrome de piernas inquietas","hipertenso","enfermedad cardíaca", "hipoglucemia", "trastorno distimico", "demencias"]
    #terms = ["Narcolepsia","Neuropatía periférica","Hipertrofia ventricular izquierda","Insuficiencia cardíaca congestiva","Diabetes mellitus tipo 2","Trastorno depresivo mayor","Enfermedad de Alzheimer"]    
        
    terms = [
    "hipersomnia",
    "narcolepsia",
    "piernas",
    "inquietas",
    "neuropatia",
    "periferica",
    "hipertenso",
    "hipertrofia",
    "ventricular",
    "izquierda",    
    "cardiaca",
    "insuficiencia",
    "congestiva",
    "hipoglucemia",
    "diabetes",
    "mellitus",
    "tipo",
    "2",
    "trastorno",
    "distimico",
    "depresivo",
    "mayor",
    "demencias",
    "alzheimer",
    "hipertension",
    "obesidad"]
    
    
    # Buscar coincidencias
    matches = find_matching_sequences(X_train, terms, vocabulary, labels)

    # Ordenar por match_count (índice 2 de la tupla), descendente
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

    # Mostrar resultados
    print("Resumen de coincidencias:")
    for idx, label, count, words in sorted_matches:
        if label == 1:
            print(f"Sec {idx} Label: {label}: {count} coincidencias → {set(words)}")
if __name__ == "__main__":
    main()