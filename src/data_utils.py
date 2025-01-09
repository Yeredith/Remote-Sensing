<<<<<<< HEAD
=======

>>>>>>> ef3cf9e75d1c1da1797a6d08e5d3a5d4c332c368
import torch
import torch.utils.data as data
import numpy as np
import os
from os.path import join
import scipy.io as scio

<<<<<<< HEAD
def is_image_file(filename):
    """Función para comprobar si el archivo es una imagen con extensión .mat"""
    return any(filename.endswith(extension) for extension in ['.mat'])

class TestsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, image_bands=None):
        """Inicializa el dataset de test desde un directorio"""
        super(TestsetFromFolder, self).__init__()
        self.image_bands = image_bands  # Guarda el parámetro image_bands (si es necesario)
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        """Carga una imagen y su etiqueta (HR) desde el archivo .mat"""
        mat = scio.loadmat(self.image_filenames[index])
        input = mat['LR'].astype(np.float32)
        label = mat['HR'].astype(np.float32)
        
        # Convertir las imágenes a formato CxHxW (canal x alto x ancho)
        input = input.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        label = label.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        
        image_name = os.path.basename(self.image_filenames[index])  # Obtener el nombre del archivo
        return torch.from_numpy(input).float(), torch.from_numpy(label).float(), image_name  # Incluye el nombre de la imagen

    def __len__(self):
        """Devuelve el número total de archivos en el dataset"""
        return len(self.image_filenames)



class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, image_bands=None):
        """
        Inicializa el dataset de entrenamiento desde un directorio.
        Puedes pasar `image_bands` si lo necesitas para algún procesamiento adicional.

        :param dataset_dir: Ruta al directorio con las imágenes
        :param image_bands: Opcional, información adicional de las bandas de la imagen (por ejemplo, si usas imágenes multicanal)
        """
        super(TrainsetFromFolder, self).__init__()
        self.image_bands = image_bands  # Almacena la información sobre las bandas si es necesario
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        """Carga una imagen y su etiqueta (HR) desde el archivo .mat"""
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        input = mat['lr'].astype(np.float32)
        label = mat['hr'].astype(np.float32)
        
        # Convertir las imágenes a formato CxHxW (canal x alto x ancho)
        input = input.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        label = label.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        
        return torch.from_numpy(input), torch.from_numpy(label)

    def __len__(self):
        """Devuelve el número total de archivos en el dataset"""
        return len(self.image_filenames)



class ValsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, image_bands=None):
        """
        Inicializa el dataset de validación desde un directorio.
        Puedes pasar `image_bands` si lo necesitas para algún procesamiento adicional.

        :param dataset_dir: Ruta al directorio con las imágenes
        :param image_bands: Opcional, información adicional de las bandas de la imagen
        """
        super(ValsetFromFolder, self).__init__()
        self.image_bands = image_bands  # Almacena la información sobre las bandas si es necesario
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        """Carga una imagen y su etiqueta (HR) desde el archivo .mat"""
        mat = scio.loadmat(self.image_filenames[index])
        input = mat['lr'].astype(np.float32)
        label = mat['hr'].astype(np.float32)
        
        # Convertir las imágenes a formato CxHxW (canal x alto x ancho)
        input = input.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        label = label.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        
        return torch.from_numpy(input).float(), torch.from_numpy(label).float()

    def __len__(self):
        """Devuelve el número total de archivos en el dataset"""
        return len(self.image_filenames)

class ValsetFromFolder2(data.Dataset):
    def __init__(self, dataset_dir, image_bands=None):
        """
        Inicializa el dataset de validación desde un directorio.
        Puedes pasar `image_bands` si lo necesitas para algún procesamiento adicional.

        :param dataset_dir: Ruta al directorio con las imágenes
        :param image_bands: Opcional, información adicional de las bandas de la imagen
        """
        super(ValsetFromFolder2, self).__init__()
        self.image_bands = image_bands  # Almacena la información sobre las bandas si es necesario
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        """Carga una imagen y su etiqueta (HR) desde el archivo .mat"""
        mat = scio.loadmat(self.image_filenames[index])
        input = mat['LR'].astype(np.float32)
        label = mat['HR'].astype(np.float32)
        
        # Convertir las imágenes a formato CxHxW (canal x alto x ancho)
        input = input.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        label = label.transpose(2, 0, 1)  # Cambiar de HxWxC a CxHxW
        
        return torch.from_numpy(input).float(), torch.from_numpy(label).float()

    def __len__(self):
        """Devuelve el número total de archivos en el dataset"""
=======

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

class TestsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TestsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index])
        input = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        label = mat['HR'].astype(np.float32).transpose(2, 0, 1)
        image_name = os.path.basename(self.image_filenames[index])  # Obtén el nombre del archivo
        return torch.from_numpy(input).float(), torch.from_numpy(label).float(), image_name  # Incluye el nombre

    def __len__(self):
        return len(self.image_filenames)


class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TrainsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        input = mat['lr'].astype(np.float32)
        label = mat['hr'].astype(np.float32)
        return torch.from_numpy(input), torch.from_numpy(label)

    def __len__(self):
        return len(self.image_filenames)


class ValsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(ValsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index])
        input = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        label = mat['HR'].astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(input).float(), torch.from_numpy(label).float()

    def __len__(self):
>>>>>>> ef3cf9e75d1c1da1797a6d08e5d3a5d4c332c368
        return len(self.image_filenames)
