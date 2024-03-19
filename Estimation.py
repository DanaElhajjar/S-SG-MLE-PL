
# --------------------------------------------------------------------------------------------------
# import librairies 
# --------------------------------------------------------------------------------------------------
import numpy as np

# --------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------
def SCM(X):  
    """ A function that computes the ML Estimator for covariance matrix estimation for gaussian data
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
        Outputs:
            * sigma_mle = the ML estimate"""
    p = X.shape[0]
    n = X.shape[1] 

    # initialization
    sigma_mle = np.zeros((p, p)) 
  
    sigma_mle = (X@X.conj().T) / n 
return sigma_mle

def tyler_estimator_covariance(X, tol=0.001, iter_max=100):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimated covariance matrix
            * τ = the estimated tau
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    stop_cond = np.inf # Distance between 2 iterations
    Sigma = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (stop_cond>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ_est = np.diagonal(X.conj().T@inv(Sigma)@X)
        X_bis = X / np.sqrt(τ_est)
        Sigma_new = (p/N) * X_bis@X_bis.conj().T

        # Imposing trace constraint: Tr(Sigma) = p
        Sigma_new = p*Sigma_new/np.trace(Sigma_new)

        # Condition for stopping
        stop_cond = np.linalg.norm(Sigma_new - Sigma, 'fro')/ np.linalg.norm(Sigma, 'fro')

        iteration = iteration + 1

        # Updating Sigma
        Sigma = Sigma_new

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')
    return Sigma, iteration

def phasecorrection3(covmatrix): 
    """
    A function that corrects the phase of a covariance matrix to ensure phase alignment with the first element.
        Inputs:
            * covmatrix : Covariance matrix representing the phase information.
        Outputs:
            * phase_minus0 : corrected phases ensuring alignment with the phase of the first element. """
    phase = -np.angle(covmatrix[0,:])
    phase_minus0 = phase-phase[0]
    return phase_minus0

def phasecorrection4(covmatrix,sigmamatrix):
    """    
    A function that corrects the phase of a covariance matrix based on the sub-diagonal elements and sigma matrix.
        Inputs:
            * covmatrix : Covariance matrix 
            * sigmamatrix : Coherence matrix
        Outputs:
        * phase : corrected phases based on sub-diagonal elements and coherence matrix. """
    subdiagphase = np.zeros((sigmamatrix.shape[0]-1))
    phase = np.zeros((sigmamatrix.shape[0]))
    for i in range (sigmamatrix.shape[0]-1):
        subdiagphase[i] = ((np.angle(covmatrix[i,i+1])) +np.pi)%(2*np.pi)-np.pi
        phase[i+1] = (phase[i]+subdiagphase[i]+np.pi)%(2*np.pi)-np.pi
    return -phase
