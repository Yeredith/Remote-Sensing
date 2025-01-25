import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models import SFCSR, MCNet, Propuesto, Modificacion1, Modificacion2
import h5py
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Rutas de datos y resultados
HR_PATH = "F:/Remote Sensing/Data/Test"
RESULTS_PATH = "F:/Remote Sensing/Data/output"
MODELS = ["SFCSR", "MCNet", "Propuesto", "Modificacion1", "Modificacion2"]
FILENAMES = [
    "L18_112648_217048_s005",
    "L18_112656_217048_s024",
    "L18_112472_217096_s001"
]

# Cargar imágenes SR y HR desde HDF5
def load_images(model_path, hr_path, filename):
    sr_file = os.path.join(model_path, f"{filename}_output.h5")
    hr_file = os.path.join(hr_path, "normal", f"{filename}.h5")

    # Leer SR desde archivo HDF5
    with h5py.File(sr_file, "r") as sr_h5:
        sr_data = sr_h5["SR"][:].astype(np.float32)
    sr_data = np.clip(sr_data, 0, 1)  # Clipping para asegurar rango válido

    # Leer HR desde archivo HDF5
    with h5py.File(hr_file, "r") as hr_h5:
        hr_data = hr_h5["HR"][:].astype(np.float32)
    hr_data = np.clip(hr_data, 0, 1)  # Normalizar HR si es necesario

    return sr_data, hr_data

# Calcular el error
def compute_error(sr_image, hr_image):
    error_image = np.abs(sr_image - hr_image).astype(np.float32)
    return error_image

# Recortar imágenes y añadir recuadro rojo
def crop_and_highlight(image, crop_coords):
    x, y, w, h = crop_coords
    cropped = image[y:y+h, x:x+w, :]
    highlighted = image.copy()

    # Dibujar recuadro rojo
    highlighted[y:y+h, x:x+2, :] = [1, 0, 0]  # Borde izquierdo
    highlighted[y:y+h, x+w-2:x+w, :] = [1, 0, 0]  # Borde derecho
    highlighted[y:y+2, x:x+w, :] = [1, 0, 0]  # Borde superior
    highlighted[y+h-2:y+h, x:x+w, :] = [1, 0, 0]  # Borde inferior

    return cropped, highlighted

def error_to_rgb(error_image):
    cmap = cm.jet
    norm = Normalize(vmin=0, vmax=np.max(error_image))
    error_mapped = cmap(norm(error_image))[:, :, :3]  # Tomar solo los primeros 3 canales (RGB)
    return (error_mapped * 255).astype(np.uint8)

# Visualizar resultados
def plot_results(filename, hr_image, results, crop_coords):
    fig, axs = plt.subplots(2, len(results) + 1, figsize=(18, 8))

    # Recortar y resaltar HR
    cropped_hr, highlighted_hr = crop_and_highlight(hr_image, crop_coords)

    # Mostrar HR en la primera columna
    axs[0, 0].imshow(highlighted_hr)
    axs[0, 0].set_title("HR (Highlighted)", fontsize=12)
    axs[0, 0].axis("off")

    axs[1, 0].imshow(cropped_hr)
    axs[1, 0].set_title("HR (Cropped)", fontsize=12)
    axs[1, 0].axis("off")

    # Mostrar resultados por modelo
    for idx, (model_name, sr_image, error_image, metrics) in enumerate(results):
        cropped_sr, _ = crop_and_highlight(sr_image, crop_coords)
        cropped_error = compute_error(cropped_sr, cropped_hr)

        # Convertir el mapa de error a RGB
        error_rgb = error_to_rgb(cropped_error)

        axs[0, idx + 1].imshow(cropped_sr)
        axs[0, idx + 1].set_title(f"{model_name} SR", fontsize=10)
        axs[0, idx + 1].axis("off")

        axs[1, idx + 1].imshow(error_rgb)
        axs[1, idx + 1].set_title(f"{model_name} Error (Cropped)", fontsize=10)
        axs[1, idx + 1].axis("off")

    plt.suptitle(f"Results for {filename}", fontsize=20)
    plt.tight_layout()
    plt.show()

# Cargar métricas
def load_metrics(metrics_path):
    metrics_file = os.path.join(metrics_path, "test_results_normal", "metrics.none")
    if not os.path.exists(metrics_file):
        return {}
    metrics_df = pd.read_csv(metrics_file)

    # Verificar que las columnas necesarias existan
    required_columns = {"Filename", "PSNR", "SSIM", "EPI"}
    if not required_columns.issubset(metrics_df.columns):
        return {}

    # Ajustar los nombres de archivo si es necesario
    return metrics_df.set_index("Filename").to_dict("index")

# Script principal
def main():
    crop_coords = (100, 100, 150, 150)  # Coordenadas de recorte (x, y, ancho, alto)

    for filename in FILENAMES:
        results = []
        hr_image = None

        for model_name in MODELS:
            model_path = os.path.join(RESULTS_PATH, model_name, "test_results_normal")
            metrics = load_metrics(os.path.join(RESULTS_PATH, model_name))

            # Cargar imágenes
            sr_image, hr_image = load_images(model_path, HR_PATH, filename)

            # Calcular imagen de error
            error_image = compute_error(sr_image, hr_image)

            # Extraer métricas para el archivo
            file_metrics = metrics.get(f"{filename}.h5", {"PSNR": 0, "SSIM": 0, "EPI": 0})

            # Agregar resultados a la lista
            results.append((model_name, sr_image, error_image, file_metrics))

        # Graficar resultados para esta imagen
        plot_results(filename, hr_image, results, crop_coords)

if __name__ == "__main__":
    main()
