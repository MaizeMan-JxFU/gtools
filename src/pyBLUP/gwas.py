import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from joblib import Parallel, delayed # for parallel processing
import gc # garbage collection
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="invalid value encountered in subtract")
from .QC import simple_QC
from .cpu_inspect import get_process_info

class GWAS:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None,kinship:np.ndarray=None,log:bool=True):
        '''
        Fast Solve of Mixed Linear Model by Brent.
        
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param kinship: Calculation method of kinship matrix nxn
        '''
        self.log = log
        y = y.reshape(-1,1) # ensure the dim of y
        X = np.concatenate([np.ones((y.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1))
        self.D, self.S, self.Dh = np.linalg.svd(kinship + 1e-6 * np.eye(y.shape[0])) # simplify G matrix for solve inversed matrix
        del kinship
        del self.D
        self.Xcov = self.Dh@X
        self.y = self.Dh@y
        result = minimize_scalar(lambda lbd: -self._NULLREML(10**(lbd)),bounds=(-8,8),method='bounded',options={'xatol': 1e-3},)
        lbd_null = 10**(result.x[0,0])
        vg_null = np.mean(self.S)
        pve = vg_null/(vg_null+lbd_null)
        self.lbd_null = lbd_null
        self.pve = pve
        self.bounds = (np.log10(lbd_null)-2,np.log10(lbd_null)+2)
        pass
    def _NULLREML(self,lbd: float):
        '''Restricted Maximum Likelihood Estimation (REML) of NULL'''
        try:
            n,p_cov = self.Xcov.shape
            p = p_cov
            V = self.S+lbd
            V_inv = 1/V
            X_cov_snp = self.Xcov
            XTV_invX = V_inv*X_cov_snp.T @ X_cov_snp
            XTV_invy = V_inv*X_cov_snp.T @ self.y
            beta = np.linalg.solve(XTV_invX, XTV_invy)
            r = self.y - X_cov_snp@beta
            rTV_invr = V_inv * r.T@r
            log_detV = np.sum(np.log(V))
            sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
            total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
            c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
            return c - total_log / 2
        except Exception as e:
            print(f"REML error: {e}, lbd={lbd}")
            return -1e8
    def _REML(self,lbd: float, snp_vec:np.array):
        '''Restricted Maximum Likelihood Estimation (REML)'''
        try:
            n,p_cov = self.Xcov.shape
            p = p_cov + 1
            V = self.S+lbd
            V_inv = 1/V
            X_cov_snp = np.column_stack([self.Xcov, snp_vec])
            XTV_invX = V_inv*X_cov_snp.T @ X_cov_snp
            XTV_invy = V_inv*X_cov_snp.T @ self.y
            try:
                beta = np.linalg.solve(XTV_invX, XTV_invy)
            except:
                beta = np.linalg.solve(XTV_invX+1e-8*np.eye(XTV_invX.shape[0]), XTV_invy)
            r = self.y - X_cov_snp@beta
            rTV_invr = V_inv * r.T@r
            log_detV = np.sum(np.log(V))
            sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
            total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX # log items
            c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2 # Constant
            return c - total_log / 2
        except:
            return -1e8
    def _fit(self,snp:np.ndarray=None):
        X = np.column_stack([self.Xcov, snp])
        n,p = X.shape
        V_inv = 1/(self.S+self.lbd_null)
        XTV_invX = V_inv*X.T@X + 1e-6*np.eye(X.shape[1])
        XTV_invy = V_inv*X.T@self.y
        beta = np.linalg.solve(XTV_invX,XTV_invy)
        r = self.y - X@beta
        rTV_invr = V_inv * r.T@r
        sigma2 = rTV_invr/(n-p)
        se = np.sqrt(np.linalg.inv(XTV_invX/sigma2)[-1,-1])
        return beta[-1,0],se
    def _HACfit(self,snp:np.ndarray=None):
        result = minimize_scalar(lambda lbd: -self._REML(10**(lbd),snp),bounds=self.bounds,method='bounded',options={'xatol': 1e-2, 'maxiter': 50},) # 寻找lbd 最大化似然函数
        lbd = self.lbd_null if not result.success else 10**(result.x[0,0])
        X = np.column_stack([self.Xcov, snp])
        n,p = X.shape
        V_inv = 1/(self.S+lbd)
        XTV_invX = V_inv*X.T@X + 1e-6*np.eye(X.shape[1])
        XTV_invy = V_inv*X.T@self.y
        beta = np.linalg.solve(XTV_invX,XTV_invy)
        r = self.y - X@beta
        rTV_invr = V_inv * r.T@r
        sigma2 = rTV_invr/(n-p)
        se = np.sqrt(np.linalg.inv(XTV_invX/sigma2)[-1,-1])
        return beta[-1,0],se,lbd
    def gwas(self,snp:np.ndarray=None,chunksize=500_000,threads=-1):
        '''
        Speed version of mlm
        
        :param snp: Marker matrix, np.ndarray, samples per rows and snp per columns
        :param chunksize: calculation number per times, int
        
        :return: beta coefficients, standard errors and p-values for each SNP, np.ndarray
        '''
        num_snp = snp.shape[1]
        chunk_indexs = [i for i in range(0,num_snp,chunksize)] # reduce the usage od memory
        chunk_indexs = chunk_indexs + [num_snp] if chunk_indexs[-1] != num_snp else chunk_indexs
        beta_se_af_p = []
        snp_retain = np.array([],dtype=bool)
        t_start = time.time()
        for ii in range(len(chunk_indexs)-1):
            gc.collect()
            snp_chunk = snp[:,chunk_indexs[ii]:chunk_indexs[ii+1]].astype(np.float32)
            snp_chunk,snp_retain_sub = simple_QC(snp_chunk)
            maf:np.ndarray = np.mean(snp_chunk,axis=0)/2
            snp_retain = np.append(snp_retain,snp_retain_sub)
            snp_chunk = self.Dh@snp_chunk
            def process_col(i):
                return self._fit(snp_chunk[:, i])
            if snp_chunk.shape[1]>0:
                results = np.array(Parallel(n_jobs=threads)(delayed(process_col)(i) for i in range(snp_chunk.shape[1])))
                beta_se_af_p.append(np.concatenate([results,maf.reshape(-1,1),2*norm.sf(np.abs(results[:,0]/results[:,1])).reshape(-1,1)],axis=1))
            if self.log:
                iter_ratio = chunk_indexs[ii+1]/num_snp
                time_cost = time.time()-t_start
                time_left = time_cost/iter_ratio
                all_time_info = f'''{round(100*iter_ratio,2)}% (time cost: {round(time_cost/60,2)}/{round(time_left/60,2)} mins)'''
                cpu,mem = get_process_info()
                print(f'''\rCPU: {cpu}%, Memory: {round(mem,2)} G, Process: {all_time_info}''',end='')
            del snp_chunk,results # 释放内存
            gc.collect()
        print()
        self.snp_retain = snp_retain
        return np.concatenate(beta_se_af_p)
    def gwasHAC(self,snp:np.ndarray=None,chunksize=500_000,threads=-1):
        '''
        Speed version of mlm
        
        :param snp: Marker matrix, np.ndarray, samples per rows and snp per columns
        :param chunksize: calculation number per times, int
        
        :return: beta coefficients, standard errors and p-values for each SNP, np.ndarray
        '''
        lbds = []
        num_snp = snp.shape[1]
        chunk_indexs = [i for i in range(0,num_snp,chunksize)] # reduce the usage od memory
        chunk_indexs = chunk_indexs + [num_snp] if chunk_indexs[-1] != num_snp else chunk_indexs
        beta_se_af_p = []
        snp_retain = np.array([],dtype=bool)
        t_start = time.time()
        for ii in range(len(chunk_indexs)-1):
            gc.collect()
            snp_chunk = snp[:,chunk_indexs[ii]:chunk_indexs[ii+1]].astype(np.float32)
            snp_chunk,snp_retain_sub = simple_QC(snp_chunk)
            maf:np.ndarray = np.mean(snp_chunk,axis=0)/2
            snp_retain = np.append(snp_retain,snp_retain_sub)
            snp_chunk = self.Dh@snp_chunk
            def process_col(i):
                '''
                solving beta and its se in multiprocess
                '''
                return self._HACfit(snp_chunk[:, i])
            if snp_chunk.shape[1]>0:
                results = np.array(Parallel(n_jobs=threads)(delayed(process_col)(i) for i in range(snp_chunk.shape[1])))
                beta_se_af_p.append(np.concatenate([results[:,:-1],maf.reshape(-1,1),2*norm.sf(np.abs(results[:,0]/results[:,1])).reshape(-1,1)],axis=1))
                lbds.extend(results[:,-1])
            if self.log:
                iter_ratio = chunk_indexs[ii+1]/num_snp
                time_cost = time.time()-t_start
                time_left = time_cost/iter_ratio
                all_time_info = f'''{round(100*iter_ratio,2)}% (time cost: {round(time_cost/60,2)}/{round(time_left/60,2)} mins)'''
                cpu,mem = get_process_info()
                print(f'''\rCPU: {cpu}%, Memory: {round(mem,2)} G, Process: {all_time_info}''',end='')
            del snp_chunk,results # release memory
            gc.collect()
        print()
        self.lbd = lbds
        self.snp_retain = snp_retain
        return np.concatenate(beta_se_af_p)
    
if __name__ == '__main__':
    pass