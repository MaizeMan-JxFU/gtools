from bioplotkit import GWASPLOT
import matplotlib.pyplot as plt
from gfreader import breader,vcfreader
import pandas as pd
import numpy as np
import argparse
import time
import socket
import logging
import sys
import os
'''
Examples:
  # Basic usage with gwasplot
  --file result.assoc.txt
  # Usage of different column names with gwasplot
  --file result.assoc.txt --chr "chr" --pos "pos" --pvalue "P_wald"
  # Usage of gwasplot and point out output path
  --file result.assoc.txt --chr "chr" --pos "pos" --pvalue "P_wald" --out test # it will saved in test.pdf

File Formats:
  file:    result.assoc.txt (txt with tab as delimiter)
        
Citation:
  https://github.com/MaizeMan-JxFU/gwasplot/
'''
def setup_logging(log_file_path):
    """set logging"""
    if os.path.exists(log_file_path) and log_file_path[-4:]=='.log':
        os.remove(log_file_path)
    # creart logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # clean exist handler
    logger.handlers.clear()
    # set log format
    formatter = logging.Formatter()
    # file handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # handler of control panel
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
def main(log:bool=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument('--file', type=str, required=True,
                               help='File of gwas results')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('--chr', type=str, default='#CHROM',
                               help='Column name of chr'
                                   '(default: %(default)s)')
    optional_group.add_argument('--pos', type=str, default='POS',
                               help='Column name of position'
                                   '(default: %(default)s)')
    optional_group.add_argument('--pvalue', type=str, default='p',
                               help='Column name of pvalue'
                                   '(default: %(default)s)')
    optional_group.add_argument('--out', type=str, default=None,
                               help='Output prefix path'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        args.file,
        args.chr,
        args.pos,
        args.pvalue,
        args.out,
    ]
    # create log file
    args.out = '.'.join(args.file.split('.')[:-1]) if args.out is None else args.out
    folder = os.path.dirname(args.out)
    if not os.path.exists(folder):
        os.mkdir(folder,0o755)
    
    logger = setup_logging(f'''{args.out}.log'''.replace('//','/'))
    logger.info('High Performance Linear Mixed Model Solver for Genome-Wide Association Studies')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("GWAS plot script")
        logger.info("*"*60)
        logger.info(f"file:          {args.file}")
        logger.info(f"chr:           {args.chr}")
        logger.info(f"pos:           {args.pos}")
        logger.info(f"pvalue:        {args.pvalue}")
        logger.info(f"output prefix: {args.out}")
        logger.info("*"*60 + "\n")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.out):
        os.makedirs(args.out, mode=0o755)
        if log:
            print(f"Created output directory: {args.out}")
    return args,logger

t_start = time.time()
args,logger = main()
file = args.file
chr_string,pos_string,pvalue_string = args.chr,args.pos,args.pvalue
df = pd.read_csv(file,sep='\t',usecols=[chr_string,pos_string,pvalue_string])
fig = plt.figure(figsize=(10,4),dpi=300)
ax =fig.add_subplot(121,)
ax2 =fig.add_subplot(122,)
plotmodel = GWASPLOT(df,chr_string,pos_string,pvalue_string,0.1)
plotmodel.manhattan(5,ax=ax)
plotmodel.qq(ax=ax2)
plt.tight_layout()
plt.savefig(f'{args.out}.pdf',)
logger.info(f'Saved in {args.out}.pdf')
lt = time.localtime()
endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
logger.info(endinfo)