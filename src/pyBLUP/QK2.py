import numpy as np
import time
from .cpu_inspect import get_process_info

class QK:
    def __init__(self,M:np.ndarray,chunksize:int=10_000,log:bool=False):
        '''
        Calculation of Q and K matrix with low memory and high speed
        
        :param M: marker matrix with n samples multiply m snp (0,1,2 int8)
        :param chunksize: int (default: 500_000)
        :param low_memory: bool (default: True)
        '''
        self.log = log
        NAmark = M<0
        miss = np.sum(NAmark,axis=0)
        M[NAmark] = 0
        del NAmark # Decrease memory usage
        maf:np.ndarray = (np.sum(M,axis=0)+1)/(2*(M.shape[0]-miss)+2)
        # Filter
        maftmark = maf>.5
        maf[maftmark] = 1 - maf[maftmark]
        M[:,maftmark] = 2 - M[:,maftmark]
        colretain = (miss/M.shape[0]<=0.05) & (maf>=0.02)
        M = M[:,colretain]
        maf = maf[colretain]
        maf = maf.astype('float32')
        
        self.maf = maf
        self.Mmean = 2*maf
        self.Mvar = 2*maf*(1-maf)
        self.M = M
        self.chunksize = chunksize
    def GRM(self,method:str=1):
        '''
        :param method: int {1-Centralization, 2-Standardization}
        :return: np.ndarray, positive definite matrix or positive semidefinite matrix
        '''
        n,m = self.M.shape
        Mvar_sum = np.sum(self.Mvar)
        Mvar_r = 1/self.Mvar
        grm = np.zeros((n,n),dtype='float32')
        t_start = time.time()
        for i in range(0,m,self.chunksize):
            i_end = min(i+self.chunksize,m)
            block_i = self.M[:,i:i_end]-self.Mmean[i:i_end]
            if method == 1:
                grm+=block_i@block_i.T/Mvar_sum
            elif method == 2:
                grm+=Mvar_r[i:i_end]/m*block_i@block_i.T
            if self.log:
                iter_ratio = i_end/m
                time_cost = time.time()-t_start
                time_left = time_cost/iter_ratio
                all_time_info = f'''{round(100*iter_ratio,2)}% (time cost: {round(time_cost/60,2)}/{round(time_left/60,2)} mins)'''
                cpu,mem = get_process_info()
                print(f'''\rCPU: {cpu}%, Memory: {round(mem,2)} G, Process of calculating GRM: {all_time_info}''',end='')
        if self.log:
            print('\nCompleted!')
        return (grm+grm.T)/2
    def PCA(self):
        '''
        random SVD
        
        :param dim: dimension of pc
        :param iter_num: iteration numbers of Q matrix
        
        :return: tuple, (eigenvec[:, :dim], eigenval[:dim])
        '''
        n,m = self.M.shape
        Mstd = np.sqrt(self.Mvar)
        MMT = np.zeros((n,n),dtype='float32')
        t_start = time.time()
        for i in range(0, m, self.chunksize):
            i_end = min(i + self.chunksize, m)
            block_i = (self.M[:, i:i_end]-self.Mmean[i:i_end])/Mstd[i:i_end]
            MMT += block_i @ block_i.T
            if self.log:
                iter_ratio = i_end/m
                time_cost = time.time()-t_start
                time_left = time_cost/iter_ratio
                all_time_info = f'''{round(100*iter_ratio,2)}% (time cost: {round(time_cost/60,2)}/{round(time_left/60,2)} mins)'''
                cpu,mem = get_process_info()
                print(f'''\rCPU: {cpu}%, Memory: {round(mem,2)} G, Process of PCA: {all_time_info}''',end='')
        if self.log:
            print('\nCompleted!')
        MMT = (MMT + MMT.T)/2
        eigval,eigvec = np.linalg.eigh(MMT/m)
        idx = np.argsort(eigval)[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        return eigvec,eigval

if __name__ == '__main__':
    from gfreader import breader
    geno = breader(r"C:\Users\82788\Desktop\Pyscript\装饰器\geno.45k")
    M = geno.iloc[:,2:].values.T
    qkmodel = QK(M,log=True)
    grm = qkmodel.GRM()
    U,S,_ = qkmodel.PCA()
    print(grm[:4,:4])
    print(U[:4,:4])
    pass
    