import numpy as np

class PCA:
    def __init__(self, dim:int=3):
        '''
        X_train: row is samples, and col is SNP
        '''
        self.dim = dim
        pass
    def fit_transfer(self, X_tain:np.ndarray, exp_min=None):
        '''
        exp_min: Minimum cumsum of variance (方差贡献率累加)
        '''
        matrix = X_tain
        p = (matrix.sum(axis=0)+1)/(2*matrix.shape[0]+2)
        matrix = (matrix - 2*p)/np.sqrt(2*p*(1-p)) 
        U, S, Vh = np.linalg.svd(matrix,full_matrices=False)
        self.egvec:np.ndarray = U
        self.egval:np.ndarray = S
        S_cum = np.cumsum(self.egval)/np.sum(self.egval)
        self.dim = np.sum(S_cum<0.01*exp_min) if exp_min is not None else self.dim
        dim = self.dim
        self.V = Vh[:dim,:].T
        return matrix@self.V # plink vec
    def pred(self,X:np.array):
        matrix = X
        p = (matrix.sum(axis=0)+1)/(2*matrix.shape[0]+2)
        matrix = (matrix - 2*p)/np.sqrt(2*p*(1-p)) 
        return matrix@self.V