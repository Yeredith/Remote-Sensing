import os
import cv2
import h5py
import numpy as np
from random import randint

upscale_factor = 4
noise_snr = 17

def add_noise_with_snr(image, target_snr):
    """Agregar ruido a una imagen basado en un SNR objetivo."""
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (target_snr / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def modcrop(img, modulo):
    """Recortar una imagen para que sea divisible por un número dado."""
    h, w = img.shape[:2]
    h = h - (h % modulo)
    w = w - (w % modulo)
    return img[:h, :w, :]

def normalize_min_max(image):
    """Normalizar la imagen entre su mínimo y máximo."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def process_test(base_dir):
    """Procesar datos de prueba (Test)."""
    test_path = os.path.join(base_dir, 'Test_HR', 'val_1st', 'HR')
    save_test_folder_normal = os.path.join(base_dir, 'Data', 'Test', 'Normal')
    save_test_folder_noise = os.path.join(base_dir, 'Data', 'Test', 'Noise')

    # Crear directorios si no existen
    os.makedirs(save_test_folder_normal, exist_ok=True)
    os.makedirs(save_test_folder_noise, exist_ok=True)

    save_test_file_normal = os.path.join(save_test_folder_normal, 'normal.h5')
    save_test_file_noise = os.path.join(save_test_folder_noise, 'noise.h5')

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró el directorio de datos de prueba: {test_path}")

    test_file_names = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    if not test_file_names:
        raise FileNotFoundError("No se encontraron archivos '.jpg' en la carpeta de datos de prueba.")

    hr_data_list = []
    lr_data_list = []
    noisy_lr_data_list = []

    for name in test_file_names:
        print(f'Procesando: {name} (Test)')
        img_path = os.path.join(test_path, name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = normalize_min_max(img)

        img = modcrop(img, upscale_factor)
        lr_img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)
        noisy_lr_img = add_noise_with_snr(lr_img, noise_snr)

        hr_data_list.append(img)
        lr_data_list.append(lr_img)
        noisy_lr_data_list.append(noisy_lr_img)

        # Guardar imágenes LR y HR en archivos HDF5 usando nombres originales
        with h5py.File(os.path.join(save_test_folder_normal, f'{os.path.splitext(name)[0]}_normal.h5'), 'w') as hf:
            hf.create_dataset('HR', data=img)
            hf.create_dataset('LR', data=lr_img)
        print(f'[INFO] Guardado normal: {name}')

        with h5py.File(os.path.join(save_test_folder_noise, f'{os.path.splitext(name)[0]}_noise.h5'), 'w') as hf:
            hf.create_dataset('HR', data=img)
            hf.create_dataset('LR', data=noisy_lr_img)
        print(f'[INFO] Guardado con ruido: {name}')

def main():
    base_dir = os.getcwd()
    process_test(base_dir)  # Procesar datos de prueba

if __name__ == '__main__':
    main()