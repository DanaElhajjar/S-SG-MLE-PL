# --------------------------------------------------------------------------------------------------
# import librairies 
# --------------------------------------------------------------------------------------------------
import numpy as np

# --------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------
def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def add_one_obs(past_matrix, new_past_line_vector, new_value):
    """ A function that adds one observation to a covariance matrix, incorporating information from a new image.
            Inputs:
                * past_matrix = covariance matrix of the past images
                * new_past_line_vector = vector of the covariance betwwen the past images and the new one
                * new_value = a scalar representing the variance of the new image

            Outputs:
                * the merged covariance matrix """
    p = past_matrix.shape[0]
    C_tilde = np.zeros((p+1, p+1), dtype=np.complex128)
    C_tilde[0:p, 0:p] = past_matrix
    C_tilde[p, 0:p] = new_past_line_vector
    C_tilde[0:p, p] = new_past_line_vector.conj().T
    C_tilde[p, p] = new_value
    return C_tilde
