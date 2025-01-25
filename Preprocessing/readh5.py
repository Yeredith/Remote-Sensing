import h5py
import numpy as np

def read_hdf5(file_path):
    """
    Leer un archivo HDF5 que contiene los datos HR y LR.

    Args:
        file_path (str): Ruta al archivo HDF5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Los datos HR (alta resolución) y LR (baja resolución).
    """
    with h5py.File(file_path, 'r') as hf:
        hr_data = np.array(hf['HR'])
        lr_data = np.array(hf['LR'])
    
    print(f"[INFO] Datos cargados desde {file_path}")
    print(f"HR Shape: {hr_data.shape}, LR Shape: {lr_data.shape}")
    return hr_data, lr_data


if __name__ == "__main__":
    file_path = r"F:\Remote Sensing\Data\Train\Normal\train_data_normal.h5"  # Ruta al archivo HDF5
    hr_data, lr_data = read_hdf5(file_path)
    
    # Ejemplo: Imprimir información del primer elemento
    print(f"Primer parche HR: {hr_data[0].shape}")
    print(f"Primer parche LR: {lr_data[0].shape}")
