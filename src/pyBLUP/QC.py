import numpy as np
    
def simple_QC(M: np.ndarray,missf = 0.05,maff = 0.02):
    '''
    return M_filtered, SNP_retain(bool array)
    '''
    M[M<0] = np.nan
    snpnan_num = np.isnan(M).sum(axis=0)
    missr = snpnan_num/M.shape[0] # missing rate
    maf = np.nanmean(M,axis=0)/2 # maf
    SNP_retain = (missr<=missf)&(maf>=maff)&(maf<=(1-maff)) # 保留缺失率低于5% maf大于2%的SNP
    M = M[:,SNP_retain]
    maf = maf[SNP_retain]
    nan_mask = np.isnan(M)
    nan_rows, nan_cols = np.where(nan_mask)
    M[nan_rows, nan_cols] = 2*maf[nan_cols]
    return M, SNP_retain