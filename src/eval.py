import math
import numpy as np
from scipy.signal import convolve2d

import pdb

def PSNR(pred, gt):
      
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))   
    
    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr

def SSIM(pred, gt):
    """Función para calcular SSIM entre dos imágenes bidimensionales (una banda)."""
    ssim = compute_ssim(pred, gt)  # Aquí pred y gt ya son imágenes 2D
    return ssim

def EPI(restored,original):
    # Verifica que las imágenes tengan el mismo número de canales
    if original.shape != restored.shape:
        raise ValueError("Las imágenes original y restaurada deben tener la misma forma")

    # Filtro Laplaciano
    H = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    epi_values = []

    # Calcula el EPI para cada canal o banda
    for channel in range(original.shape[0]):  # Itera sobre cada banda o canal
        # Aplica el filtro Laplaciano
        delta_original = convolve2d(original[channel], H, mode='same', boundary='symm')
        delta_restored = convolve2d(restored[channel], H, mode='same', boundary='symm')

        # Promedios de los resultados filtrados
        mean_delta_original = np.mean(delta_original)
        mean_delta_restored = np.mean(delta_restored)

        # Componentes de preservación de bordes
        p1 = delta_original - mean_delta_original
        p2 = delta_restored - mean_delta_restored

        # Cálculo del índice EPI para el canal actual
        num = np.sum(p1 * p2)
        den = np.sqrt(np.sum(p1**2) * np.sum(p2**2))
        epi_channel = num / den if den != 0 else 0
        epi_values.append(epi_channel)

    # Promedio del EPI sobre todos los canales
    return np.mean(epi_values)
  	


def matlab_style_gauss2D(shape=np.array([11,11]),sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    siz = (shape-np.array([1,1]))/2
    std = sigma
    eps = 2.2204e-16
    x = np.arange(-siz[1], siz[1]+1, 1)
    y = np.arange(-siz[0], siz[1]+1, 1)
    m,n = np.meshgrid(x, y)
    
    h = np.exp(-(m*m + n*n).astype(np.float32) / (2.*sigma*sigma))    
    h[ h < eps*h.max() ] = 0    	
    sumh = h.sum()   	

    if sumh != 0:
        h = h.astype(np.float32) / sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=np.array([win_size,win_size]), sigma=1.5)
    window = window.astype(np.float32)/np.sum(np.sum(window))
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)).astype(np.float32) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))
 