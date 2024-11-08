from data import *
from rd.real_data import process_images
from src.utility import writetoENVIformat

import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("MSE simulation of the phase difference")
    parser.add_argument("--l", 
                        type=int, 
                        default=20,
                        help="Number of time stamps in the time series")
    parser.add_argument("--n_trials", 
                        type=int, 
                        default=1000, 
                        help="Number of Monte-Carlo Trials")
    parser.add_argument("--phasecorrectionchoice",
                        type=int, 
                        default=4, 
                        help="Choice of the phase correction")
    parser.add_argument("--tol", 
                        type=float, 
                        default=0.001, 
                        help="Treshold for stopping the BCD algorithm")
    parser.add_argument("--model", 
                        type=str, 
                        default="ScaledGaussian", 
                        help="The model")
    parser.add_argument("--p", 
                        type=int, 
                        default=19, 
                        help="The size of the past time series")
    parser.add_argument("--rank", 
                        type=int, 
                        default=20, 
                        help="The rank of the covariance matrix")
    parser.add_argument("--iter_max_BCD", 
                        type=int, 
                        default=30, 
                        help="Number of iterations of the BCD algorithms")
    parser.add_argument("--iter_PL", 
                        type=int, 
                        default=100, 
                        help="The number of iteration of the PL algorithm")
    parser.add_argument("--iter_max_MM", 
                        type=int, 
                        default=50, 
                        help="The number of iteration of the MM algorithm")
    parser.add_argument("--Multi",
                        type=bool,
                        default=True,
                        help="Parallel computing choice")
    parser.add_argument("--n_threads", 
                        type=int, 
                        default=-1, 
                        help="The number of threads")
    parser.add_argument("--window_size",
                        type=int,
                        default=8,
                        help="The size of the window")
    args = parser.parse_args()

    # Safely parse known arguments and ignore unknown arguments passed by Jupyter
    args, unknown = parser.parse_known_args()

    # Convert `n_list` to a list of integers
    args.n_list = [int(x) for x in args.n_list.split(",")]

    # add `argsMLEPL`
    args.argsMLEPL = [args.iter_PL, args.iter_max_BCD, args.iter_max_MM, args.phasecorrectionchoice]

    print("Phase difference on real datawith parameters:")
    for key, val in vars(args).items():
        print(f"  * {key}: {val}")


if __name__ == "__main__":

    DataFolder = "./data/" 

    # interferogram size : (4188, 23887)

    output = process_images(DataFolder, args.window_size, args.n_threads, args.n, args.argsMLEPL, args.tol)

    phases_list = []
    C_tilde_list = []
    Sigma_tilde_list = []
    for t in output:
        phases_list.append(t[0])
        C_tilde_list.append(t[1])
        Sigma_tilde_list.append(t[2])

    phases_array = np.array(phases_list)
    C_tilde_array = np.array(C_tilde_list)
    Sigma_tilde_array = np.array(Sigma_tilde_list)

    lenght = len(Sigma_tilde_array)
    sigma = np.zeros(lenght)
    for i in range(lenght):
        sigma[i] = Sigma_tilde_array[i, 0, -1]

    Sigma_reshaped = sigma.reshape(609, 1913)  # tailles recuperes du hdr de coherence multilook pour une taille
    # overlap 1 pixel : (2435, 7651)
    # overlap 4 pixels : 8- (609, 1913) ((3684-1244)-8) / overlap 3 pixels : # 6- (812, 2551)

    output_filename = "Sigma_sequential_date_20_process=5_windowsize=8.img"
    writetoENVIformat(output_filename, Sigma_reshaped)

    estimated_inter = phases_array.reshape(609, 1913)   # 6- (698, 3981) 7- (598, 3412) 5- (837, 4777) 8- (523, 2985) # crop : (305,957)

    plt.imshow(estimated_inter, plt.get_cmap('jet'))
    plt.colorbar()

    output_filename = "phase_sequential_date_20_process=5_windowsize=8.img"
    writetoENVIformat(output_filename, estimated_inter)