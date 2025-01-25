import os
import cv2
import h5py
import numpy as np
import gc

def augment_patches(hr_patch, lr_patch):
    yield hr_patch, lr_patch  # Original

    # Rotar 90
    hr_90 = np.rot90(hr_patch, k=1, axes=(0, 1))
    lr_90 = np.rot90(lr_patch, k=1, axes=(0, 1))
    yield hr_90, lr_90

    # Flip vertical
    hr_flipud = np.flipud(hr_patch)
    lr_flipud = np.flipud(lr_patch)
    yield hr_flipud, lr_flipud

    # Flip horizontal
    hr_fliplr = np.fliplr(hr_patch)
    lr_fliplr = np.fliplr(lr_patch)
    yield hr_fliplr, lr_fliplr

def add_noise_with_snr(image, snr):

    signal_power = np.mean(image ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0.0, 1.0)
    return noisy_image

def save_batch_to_h5(hr_data_list, lr_data_list, save_path,
                     idx, patch_size_hr, patch_size_lr):

    mode = 'a' if os.path.exists(save_path) else 'w'
    with h5py.File(save_path, mode) as hf:
        if 'HR' not in hf:
            hf.create_dataset(
                'HR',
                data=np.concatenate(hr_data_list, axis=0),
                maxshape=(None, patch_size_hr, patch_size_hr, 3),
                chunks=True
            )
            hf.create_dataset(
                'LR',
                data=np.concatenate(lr_data_list, axis=0),
                maxshape=(None, patch_size_lr, patch_size_lr, 3),
                chunks=True
            )
        else:
            hf['HR'].resize(hf['HR'].shape[0] + len(hr_data_list), axis=0)
            hf['HR'][-len(hr_data_list):] = np.concatenate(hr_data_list, axis=0)

            hf['LR'].resize(hf['LR'].shape[0] + len(lr_data_list), axis=0)
            hf['LR'][-len(lr_data_list):] = np.concatenate(lr_data_list, axis=0)

    print(f"Guardado lote con {len(hr_data_list)} elementos. Total procesado: {idx}")

def prepare_data_rgb_val(base_path, save_path, 
                         patch_size_hr=128, patch_size_lr=32,
                         random_number=10, scale=4,
                         start_index=3238, end_index=4046, 
                         batch_size=500):

    file_names = sorted([f for f in os.listdir(base_path) if f.endswith(('.jpg', '.png'))])
    file_names = file_names[start_index - 1:end_index]

    num_images = len(file_names)
    print(f"Procesando {num_images} imágenes para VALIDACIÓN desde el índice {start_index} hasta {end_index}")

    hr_data_list = []
    lr_data_list = []
    idx = 0

    for i, file_name in enumerate(file_names):
        print(f'Procesando: {file_name} ({i+1}/{num_images})')
        img_path = os.path.join(base_path, file_name)

        hr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if hr_img is None:
            print(f"No se pudo leer la imagen: {file_name}")
            continue

        h, w, _ = hr_img.shape
        if h < patch_size_hr or w < patch_size_hr:
            print(f"La imagen {file_name} es demasiado pequeña para el parche HR {patch_size_hr}x{patch_size_hr}.")
            continue

        lr_img = cv2.resize(hr_img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

        for _ in range(random_number):
            x_lr = np.random.randint(0, lr_img.shape[0] - patch_size_lr)
            y_lr = np.random.randint(0, lr_img.shape[1] - patch_size_lr)

            x_hr = x_lr * scale
            y_hr = y_lr * scale

            hr_patch = hr_img[x_hr:x_hr + patch_size_hr, y_hr:y_hr + patch_size_hr, :]
            lr_patch = lr_img[x_lr:x_lr + patch_size_lr, y_lr:y_lr + patch_size_lr, :]

            # Validar dimensiones
            if hr_patch.shape[0] != patch_size_hr or hr_patch.shape[1] != patch_size_hr:
                continue
            if lr_patch.shape[0] != patch_size_lr or lr_patch.shape[1] != patch_size_lr:
                continue

            # Data augmentation
            for hr_aug, lr_aug in augment_patches(hr_patch, lr_patch):
                # Normalizar
                hr_aug = hr_aug.astype(np.float32) / 255.0
                lr_aug = lr_aug.astype(np.float32) / 255.0

                hr_data_list.append(np.expand_dims(hr_aug, axis=0))
                lr_data_list.append(np.expand_dims(lr_aug, axis=0))

                idx += 1

            # Guardar si llegamos al batch_size
            if len(hr_data_list) >= batch_size:
                save_batch_to_h5(hr_data_list, lr_data_list, save_path,
                                 idx, patch_size_hr, patch_size_lr)
                hr_data_list.clear()
                lr_data_list.clear()
                gc.collect()

    # Guardar último lote
    if len(hr_data_list) > 0:
        save_batch_to_h5(hr_data_list, lr_data_list, save_path,
                         idx, patch_size_hr, patch_size_lr)
        print(f"Guardado el lote final (VALIDACIÓN) con {len(hr_data_list)} elementos.")


def prepare_data_with_noise_val(base_path, save_path,
                                patch_size_hr=128, patch_size_lr=32,
                                random_number=10, scale=4, snr=20,
                                start_index=3238, end_index=4046,
                                batch_size=500):

    file_names = sorted([f for f in os.listdir(base_path) if f.endswith(('.jpg', '.png'))])
    file_names = file_names[start_index - 1:end_index]

    num_images = len(file_names)
    print(f"Procesando {num_images} imágenes para VALIDACIÓN (con ruido) desde {start_index} hasta {end_index}")

    hr_data_list = []
    lr_data_list = []
    idx = 0

    for i, file_name in enumerate(file_names):
        print(f'Procesando: {file_name} ({i+1}/{num_images})')
        img_path = os.path.join(base_path, file_name)

        hr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if hr_img is None:
            print(f"No se pudo leer la imagen: {file_name}")
            continue

        h, w, _ = hr_img.shape
        if h < patch_size_hr or w < patch_size_hr:
            print(f"La imagen {file_name} es demasiado pequeña para {patch_size_hr}x{patch_size_hr}.")
            continue

        lr_img = cv2.resize(hr_img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

        for _ in range(random_number):
            x_lr = np.random.randint(0, lr_img.shape[0] - patch_size_lr)
            y_lr = np.random.randint(0, lr_img.shape[1] - patch_size_lr)

            x_hr = x_lr * scale
            y_hr = y_lr * scale

            hr_patch = hr_img[x_hr:x_hr + patch_size_hr, y_hr:y_hr + patch_size_hr, :]
            lr_patch = lr_img[x_lr:x_lr + patch_size_lr, y_lr:y_lr + patch_size_lr, :]

            if hr_patch.shape[0] != patch_size_hr or hr_patch.shape[1] != patch_size_hr:
                continue
            if lr_patch.shape[0] != patch_size_lr or lr_patch.shape[1] != patch_size_lr:
                continue

            # Normalizar HR y LR antes de añadir ruido
            hr_patch = hr_patch.astype(np.float32) / 255.0
            lr_patch = lr_patch.astype(np.float32) / 255.0

            # Añadir ruido
            lr_patch_noisy = add_noise_with_snr(lr_patch, snr)

            # Data augmentation
            for hr_aug, lr_aug in augment_patches(hr_patch, lr_patch_noisy):
                hr_data_list.append(np.expand_dims(hr_aug, axis=0))
                lr_data_list.append(np.expand_dims(lr_aug, axis=0))

                idx += 1

            # Guardar si llegamos al batch_size
            if len(hr_data_list) >= batch_size:
                save_batch_to_h5(hr_data_list, lr_data_list, save_path,
                                 idx, patch_size_hr, patch_size_lr)
                hr_data_list.clear()
                lr_data_list.clear()
                gc.collect()

    # Guardar último lote
    if len(hr_data_list) > 0:
        save_batch_to_h5(hr_data_list, lr_data_list, save_path,
                         idx, patch_size_hr, patch_size_lr)
        print(f"Guardado el lote final (VALIDACIÓN ruido) con {len(hr_data_list)} elementos.")


if __name__ == "__main__":
    root_dir = os.getcwd()
    base_path = os.path.join(root_dir, "Train_HR_data")

    # Rutas donde guardar datos de validación
    data_val_dir_normal = os.path.join(root_dir, "Data", "Validation", "Normal")
    data_val_dir_noise = os.path.join(root_dir, "Data", "Validation", "Noise")
    os.makedirs(data_val_dir_normal, exist_ok=True)
    os.makedirs(data_val_dir_noise, exist_ok=True)

    save_path_val_normal = os.path.join(data_val_dir_normal, "validation_data_normal.h5")
    save_path_val_noise = os.path.join(data_val_dir_noise, "validation_data_noise.h5")

    # Generar datos de validación sin ruido (con data augmentation)
    prepare_data_rgb_val(
        base_path=base_path,
        save_path=save_path_val_normal,
        patch_size_hr=128,
        patch_size_lr=32,
        random_number=4,
        scale=4,
        start_index=3238,
        end_index=4046,
        batch_size=500
    )

    # Generar datos de validación con ruido (con data augmentation)
    prepare_data_with_noise_val(
        base_path=base_path,
        save_path=save_path_val_noise,
        patch_size_hr=128,
        patch_size_lr=32,
        random_number=4,
        scale=4,
        snr=17,
        start_index=3238,
        end_index=4046,
        batch_size=500
    )
