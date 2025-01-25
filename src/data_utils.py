import torch
import torch.utils.data as data
import os
from os.path import join
import h5py
import numpy as np

def is_h5_file(filename):
    """Verifica si el archivo tiene extensión .h5"""
    return filename.endswith('.h5')

class TestsetFromFolder(data.Dataset):
    """
    Lee cada imagen de test (normal) desde un archivo .h5 individual.
    Cada .h5 en dataset_dir debe contener 2 datasets: 'LR' y 'HR'.
    Dimensiones típicas:
        HR: (480, 480, 3)
        LR: (120, 120, 3)
    """
    def __init__(self, dataset_dir, image_bands=None):
        super(TestsetFromFolder, self).__init__()
        self.image_bands = image_bands
        self.image_filenames = [
            join(dataset_dir, x)
            for x in os.listdir(dataset_dir) if is_h5_file(x)
        ]

    def __getitem__(self, index):
        h5_path = self.image_filenames[index]
        # Cargamos el archivo .h5 (un solo ejemplo: LR, HR)
        with h5py.File(h5_path, 'r') as hf:
            lr_data = hf['LR'][:]   # (H_lr, W_lr, 3)
            hr_data = hf['HR'][:]   # (H_hr, W_hr, 3)

        # Transponemos a (3, H, W)
        lr_data = lr_data.transpose(2, 0, 1)
        hr_data = hr_data.transpose(2, 0, 1)

        image_name = os.path.basename(h5_path)
        return (
            torch.from_numpy(lr_data).float(),
            torch.from_numpy(hr_data).float(),
            image_name
        )

    def __len__(self):
        return len(self.image_filenames)


class TestsetFromFolder_Noisy(data.Dataset):
    """
    Lee cada imagen de test (ruidosa) desde un archivo .h5 individual.
    Cada .h5 en dataset_dir debe contener 2 datasets: 'LR' (con ruido) y 'HR'.
    Dimensiones típicas:
        HR: (480, 480, 3)
        LR: (120, 120, 3)  [con ruido]
    """
    def __init__(self, dataset_dir, image_bands=None):
        super(TestsetFromFolder_Noisy, self).__init__()
        self.image_bands = image_bands
        self.image_filenames = [
            join(dataset_dir, x)
            for x in os.listdir(dataset_dir) if is_h5_file(x)
        ]

    def __getitem__(self, index):
        h5_path = self.image_filenames[index]
        with h5py.File(h5_path, 'r') as hf:
            # 'LR' aquí en realidad es la imagen con ruido (según tu generate_data_test.py)
            lr_data = hf['LR'][:]   # (H_lr, W_lr, 3)
            hr_data = hf['HR'][:]   # (H_hr, W_hr, 3)

        lr_data = lr_data.transpose(2, 0, 1)
        hr_data = hr_data.transpose(2, 0, 1)

        image_name = os.path.basename(h5_path)
        return (
            torch.from_numpy(lr_data).float(),
            torch.from_numpy(hr_data).float(),
            image_name
        )

    def __len__(self):
        return len(self.image_filenames)


class TrainsetFromFolder(data.Dataset):
    """
    Lee los datos de entrenamiento (normal o ruido) desde UN archivo .h5.
    El .h5 debe contener:
        'LR': (n, 32, 32, 3)
        'HR': (n, 128, 128, 3)
    """
    def __init__(self, dataset_dir, image_bands=None):
        super(TrainsetFromFolder, self).__init__()
        self.image_bands = image_bands
        # Se asume que solo hay UN .h5 en dataset_dir
        h5_files = [
            join(dataset_dir, x)
            for x in os.listdir(dataset_dir) if is_h5_file(x)
        ]
        if len(h5_files) != 1:
            raise ValueError(f"Se esperaba exactamente 1 archivo .h5 en {dataset_dir}, encontrado {len(h5_files)}.")
        self.h5_path = h5_files[0]
        # Abrimos para leer la estructura
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.lr_data = self.h5_file['LR']  # shape (n, H_lr, W_lr, 3)
        self.hr_data = self.h5_file['HR']  # shape (n, H_hr, W_hr, 3)
        self.n_items = self.lr_data.shape[0]

    def __getitem__(self, index):
        lr_patch = self.lr_data[index]  # (H_lr, W_lr, 3)
        hr_patch = self.hr_data[index]  # (H_hr, W_hr, 3)

        # Transponer a (3, H, W)
        lr_patch = lr_patch.transpose(2, 0, 1)
        hr_patch = hr_patch.transpose(2, 0, 1)

        lr_tensor = torch.from_numpy(lr_patch).float()
        hr_tensor = torch.from_numpy(hr_patch).float()
        return lr_tensor, hr_tensor

    def __len__(self):
        return self.n_items


class ValsetFromFolder(data.Dataset):
    """
    Lee los datos de validación (normal o ruido) desde UN archivo .h5.
    El .h5 debe contener:
        'LR': (n, H_lr, W_lr, 3)
        'HR': (n, H_hr, W_hr, 3)
    """
    def __init__(self, dataset_dir, image_bands=None):
        super(ValsetFromFolder, self).__init__()
        self.image_bands = image_bands
        # Se asume que solo hay UN .h5 en dataset_dir
        h5_files = [
            join(dataset_dir, x)
            for x in os.listdir(dataset_dir) if is_h5_file(x)
        ]
        if len(h5_files) != 1:
            raise ValueError(f"Se esperaba exactamente 1 archivo .h5 en {dataset_dir}, encontrado {len(h5_files)}.")
        self.h5_path = h5_files[0]
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.lr_data = self.h5_file['LR']
        self.hr_data = self.h5_file['HR']
        self.n_items = self.lr_data.shape[0]

    def __getitem__(self, index):
        lr_patch = self.lr_data[index]  # (H_lr, W_lr, 3)
        hr_patch = self.hr_data[index]  # (H_hr, W_hr, 3)

        lr_patch = lr_patch.transpose(2, 0, 1)
        hr_patch = hr_patch.transpose(2, 0, 1)

        return torch.from_numpy(lr_patch).float(), torch.from_numpy(hr_patch).float()

    def __len__(self):
        return self.n_items


class ValsetFromFolder2(data.Dataset):
    def __init__(self, dataset_dir, image_bands=None):
        super(ValsetFromFolder2, self).__init__()
        self.image_bands = image_bands
        h5_files = [
            join(dataset_dir, x)
            for x in os.listdir(dataset_dir) if is_h5_file(x)
        ]
        if len(h5_files) != 1:
            raise ValueError(f"Se esperaba exactamente 1 archivo .h5 en {dataset_dir}, encontrado {len(h5_files)}.")
        self.h5_path = h5_files[0]
        self.h5_file = h5py.File(self.h5_path, 'r')
        # Aquí asumimos que están en 'LR' y 'HR', igual que antes.
        self.lr_data = self.h5_file['LR']
        self.hr_data = self.h5_file['HR']
        self.n_items = self.lr_data.shape[0]

    def __getitem__(self, index):
        lr_patch = self.lr_data[index]
        hr_patch = self.hr_data[index]
        lr_patch = lr_patch.transpose(2, 0, 1)
        hr_patch = hr_patch.transpose(2, 0, 1)
        return torch.from_numpy(lr_patch).float(), torch.from_numpy(hr_patch).float()

    def __len__(self):
        return self.n_items
