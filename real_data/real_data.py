import os
import sys
import numpy as np
from osgeo import gdal
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.estimation import extractMLEfunc, tyler_estimator_covariance
from optimization import S_SG_MLE_PL_BCD

def extraire_numero_fichier(nom_fichier):
    return int(nom_fichier.split('_')[-1].split('.data')[0])

def process_window(p, n, rank, argsMLEPL, tol, ifg_window):
    """ A function that apply the approach on a window

    Args:
        p (int): length of the time series of past images
        n (int): sample size
        rank (int): rank of the covariance matrix
        argsMLEPL (tuple): tuple of number of iterations
        tol (float): treshold for stopping the BCD algorithm
        ifg_window (array): sample 

    Returns:
        new_phase_sequential : the value of the phase of the new image
        C_tilde_sequential : estimated covariance matrix
        Sigma_tilde_sequential : estimated coherence matrix
    """
    (iter_max_BCD, iter_max_MM, phasecorrectionchoice) = argsMLEPL
    X = np.zeros((p + 1, n), dtype=np.complex128) # initialization of the sample array
    # Parcourir les listes principales
    for j in range(ifg_window.shape[-1]):  # to iterate over the list of samples (number of interferograms) 
        X[j, :] = np.ravel(ifg_window[:, :, j]) # Add the element from index window_idx to X.
    X = X + 0.0000001 # to avoid zero pixels
    X_past = X[0:p, :] # array of the past images
    x_newdata = X[p, :] # array of the new image
    C, _ = tyler_estimator_covariance(X_past, 0.001, 100) # covariance matrix of the past
    # C = SCM(X_past) # covariance matrix of the past
    C = np.asarray(C, dtype=np.complex128)
    # MLE-PL approach on the dataset of past images
    _, _, diag_w_past, _ = extractMLEfunc(X_past, 'ScaledGaussian', 'Cor-Arg', rank, argsMLEPL)
    # S-MLE-PL approach on the dataset of the new image
    C_tilde_sequential, Sigma_tilde_sequential, new_phase_sequential = S_SG_MLE_PL_BCD(X, X_past, x_newdata, C, diag_w_past, iter_max_BCD, phasecorrectionchoice, tol)
    return new_phase_sequential, C_tilde_sequential, Sigma_tilde_sequential


def process_images(DataFolder, window_size, num_processes, argsMLEPL, tol):
    """A function that apply the process on the entire image

    Args:
        DataFolder (str): path to the data  folder
        window_size (int): size of the window
        num_processes (int): number of processes
        argsMLEPL (tuple): tuple of number of iterations
        tol (float): treshold for stopping the BCD algorithm

    Returns:
        output (tuple): tuple of process_window outputs
    """
    n = window_size**2 # sample size
    # To retrieve the .data folders of the images
    List_data = [f for f in os.listdir(DataFolder) if f.endswith('.data')]

    # Sort the list of files using the sorting function 
    List_data_sorted = sorted(List_data, key=extraire_numero_fichier)

    # Retrieve the real and imaginary parts of each image file
    List_real_components = [f for i in List_data_sorted for f in os.listdir(os.path.join(DataFolder, i)) if f.startswith('i') and f.endswith('.img')]
    List_immaginary_components = [f for i in List_data_sorted for f in os.listdir(os.path.join(DataFolder, i)) if f.startswith('q') and f.endswith('.img')]
    # list of paths
    List_paths = [DataFolder+f for f in List_data_sorted]
    # Reading the real and imaginary files of each image using GDAL
    List_real_gdal = [gdal.Open(os.path.join(path, f)) for path in List_paths for f in List_real_components if os.path.exists(os.path.join(path, f))]
    List_immaginary_gdal = [gdal.Open(os.path.join(path, f)) for path in List_paths for f in List_immaginary_components if os.path.exists(os.path.join(path, f))]
    # conversion to array
    List_real_array = [f.ReadAsArray() for f in List_real_gdal]
    List_immaginary_array = [f.ReadAsArray() for f in List_immaginary_gdal]
    List_inter = [List_real_array[i] + 1j * List_immaginary_array[i] for i in range(len(List_data_sorted))]

    Array_inter = np.array(List_inter)
    Array_inter = Array_inter.transpose(1, 2, 0)

    array_inter_crop = Array_inter[1244:3684, 13200:20856, :] # [:, 476:3860, 11948:22460]
    Array_inter = array_inter_crop

    ifg_window_time_series = []

    overlap = window_size/2
    for y in range(0, Array_inter.shape[0] - window_size + 1, overlap): # Traverse the x-axis of the window.
        for x in range(0, Array_inter.shape[1] - window_size + 1, overlap): # Traverse the y-axis of the window
            window = Array_inter[y:y+window_size, x:x+window_size] # Extraction of the image region covered by the sliding window    
            ifg_window_time_series.append(window) # len = 1561155 : window number on each image
        
    p = ifg_window_time_series[0].shape[-1] - 1
    rank = p + 1

    process_window_partial = partial(process_window, p, n, rank, argsMLEPL, tol)

    output = []
    with Pool(num_processes) as pool:
        for results in pool.imap(process_window_partial, ifg_window_time_series):
            output.append(results)
    
    return output