import numpy as np
import time
from .QC import simple_QC
import gc
class QK:
    def __init__(self,M:np.ndarray,chunksize:int=500_000,low_memory: bool=True,log:bool=False):
        '''
        Calculation of Q and K matrix with low memory and high speed
        
        :param M: marker matrix with n samples multiply m snp (0,1,2 int8)
        :param chunksize: int (default: 500_000)
        :param low_memory: bool (default: True)
        '''
        self.log = log
        n,m = M.shape
        self.chunksize = chunksize
        
        self.p_i = np.zeros(m, dtype=np.float32)
        self.Mmean = np.zeros(m, dtype=np.float32)
        self.Mstd = np.zeros(m, dtype=np.float32)
        if low_memory:
            self.M = np.zeros((n,m), dtype=np.int8)
        else:
            self.M = np.zeros((n,m), dtype=np.float32)
        retained_snps = 0
        
        t_start = time.time()
        for ii in range(0,m,chunksize):
            gc.collect()
            end_idx = min(ii + chunksize, m)
            M_chunk = M[:,ii:end_idx].astype(np.float32,copy=False)
            M_chunk,SNP_retain_sub = simple_QC(M_chunk)
            chunk_size_retained = sum(SNP_retain_sub)
            if chunk_size_retained > 0:
                idx_slice = slice(retained_snps, retained_snps + chunk_size_retained)
                self.p_i[idx_slice] = (M_chunk.sum(axis=0)+1)/(2*n+2)
                self.Mmean[idx_slice] = M_chunk.mean(axis=0)
                self.Mstd[idx_slice] = M_chunk.std(axis=0)
                self.M[:,idx_slice] = M_chunk
                retained_snps += chunk_size_retained
            if self.log:
                iter_ratio = end_idx/m
                time_cost = time.time()-t_start
                time_left = time_cost/iter_ratio
                all_time_info = f'''{round(100*iter_ratio,2)}% (time cost: {round(time_cost/60,2)}/{round(time_left/60,2)} mins)'''
                print(f'''\rInitialization of loading and QC: {all_time_info}''',end='')
            del M_chunk
            gc.collect()
        if self.log:
            print()
        del M
        gc.collect()
        # 截断数组到实际大小
        self.p_i = self.p_i[:retained_snps]
        self.Mmean = self.Mmean[:retained_snps]
        self.Mstd = self.Mstd[:retained_snps]
        self.std = np.sqrt(2 * self.p_i * (1 - self.p_i))
        self.M = self.M[:,:retained_snps]
    def _k(self, Msub:np.ndarray=None,method:str='VanRanden'):
        if method == 'VanRanden':
            Z:np.ndarray = Msub - 2*self.p_i
            p_sum = 2*np.sum(self.p_i*(1-self.p_i))
            return Z@Z.T/p_sum
        elif method == 'gemma1':
            Z:np.ndarray = Msub - self.Mmean
            return Z@Z.T/Z.shape[1]
        elif method == 'gemma2':
            Z:np.ndarray = (Msub - self.Mmean)/self.Mstd
            return Z@Z.T/Z.shape[1]
        elif method == 'pearson':
            return np.corrcoef(Msub)
    def kinship(self,split_num:int=10,method:str='VanRanden'):
        '''
        :param split_num: int
        :param method: str {'VanRanden', 'gemma1', 'gemma2', 'pearson'}
        :return: np.ndarray, positive definite matrix or positive semidefinite matrix
        '''
        n,m = self.M.shape
        o = int(split_num*(split_num-1)/2)
        chunks = np.linspace(0,n,split_num,dtype=int)
        kin = np.zeros(shape=(n,n),dtype=np.float32)
        iter_num = 0
        t_start = time.time()
        for ind1 in range(len(chunks)-1):
            for ind2 in range(ind1,len(chunks)-1):
                iter_num+=1
                SNP_sub = np.concatenate([self.M[chunks[ind1]:chunks[ind1+1],:],self.M[chunks[ind2]:chunks[ind2+1],:]],axis=0,dtype=np.float32) # 分块计算 kinship
                kin[chunks[ind1]:chunks[ind1+1],chunks[ind2]:chunks[ind2+1]] = self._k(SNP_sub,method)[:chunks[ind1+1]-chunks[ind1],chunks[ind1+1]-chunks[ind1]:]
                del SNP_sub
                if self.log:
                    iter_ratio = iter_num/o
                    time_cost = time.time()-t_start
                    time_left = time_cost/iter_ratio
                    all_time_info = f'''{round(100*iter_ratio,2)}% (time cost: {round(time_cost/60,2)}/{round(time_left/60,2)} mins)'''
                    print(f'''\rProgress of calculating GRM: {all_time_info}''',end='')
        if self.log:
            print()
        return np.triu(kin,k=0)+np.triu(kin,k=1).T
    def pca(self,):
        '''
        test result of rpca
        '''
        M = (self.M - 2*self.p_i)/np.sqrt(2*self.p_i*(1-self.p_i)) # standard M matrix
        eigenvec, eigenval, Vh = np.linalg.svd(M,full_matrices=False)
        return eigenvec, eigenval
    def rpca(self, dim=10, iter_num=5, chunk_size=5_000):
        '''
        random SVD
        
        :param dim: dimension of pc
        :param iter_num: iteration numbers of Q matrix
        
        :return: tuple, (eigenvec[:, :dim], eigenval[:dim])
        '''
        # M = ((self.M - 2 * self.p_i) / self.std).T  # n x m matrix
        n, m = self.M.T.shape
        l = dim + 10
        # 分块生成随机矩阵和计算 Y
        omega = np.random.normal(size=(m, l))
        Y = np.zeros((n, l), dtype=np.float32)
        # 分块计算 Y = M @ omega
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            M_sub = ((self.M[:,i:end_i] - 2 * self.p_i[i:end_i]) / self.std[i:end_i]).T
            Y[i:end_i] = M_sub @ omega
        # 幂迭代也使用分块
        t_start = time.time()
        for _ in range(iter_num):
            Q = np.linalg.qr(Y)[0]
            # 分块计算 Z = M.T @ Q
            Z = np.zeros((m, l), dtype=np.float32)
            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                M_sub = ((self.M[:,i:end_i] - 2 * self.p_i[i:end_i]) / self.std[i:end_i]).T
                Z += M_sub.T @ Q[i:end_i]
            # 分块计算 Y = M @ Z
            Y.fill(0)
            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                M_sub = ((self.M[:,i:end_i] - 2 * self.p_i[i:end_i]) / self.std[i:end_i]).T
                Y[i:end_i] = M_sub @ Z
            if self.log:
                iter_ratio = (_+1)/iter_num
                time_cost = time.time()-t_start
                time_left = time_cost/iter_ratio
                all_time_info = f'''{round(100*iter_ratio,2)}% (time cost: {round(time_cost/60,2)}/{round(time_left/60,2)} mins)'''
                print(f'''\rProgress of calculating population stratification (randomSVD, PCA): {all_time_info}''',end='')
        if self.log:
            print()
        Q, _ = np.linalg.qr(Y)
        # 分块计算 B = Q.T @ M
        B = np.zeros((l, m), dtype=np.float32)
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            M_sub = ((self.M[:,i:end_i] - 2 * self.p_i[i:end_i]) / self.std[i:end_i]).T
            B += Q[i:end_i].T @ M_sub
        _, eigenval, eigenvec = np.linalg.svd(B, full_matrices=False)
        return eigenvec.T[:, :dim], eigenval[:dim]

if __name__ == '__main__':
    pass
    