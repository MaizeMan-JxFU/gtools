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
  # Basic usage with xxx
  --param1 genotypes.vcf --param2 phenotypes.txt --param3 results

File Formats:
  xxx:    xxx (bed/bim/fam)
        
Citation:
  https://github.com/MaizeMan-JxFU/xxx/
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
    required_group.add_argument('--files', type=str, required=True,
                               help='files with tsv format, example: file1,file2,file3,...')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('--out', type=str, default='test',
                               help='Output prefix path'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    args.files = args.files.replace(' ','').split(',')
    # Create output directory if it doesn't exist
    if  '/' in args.out:
        folder = os.path.dirname(args.out)
        if not os.path.exists(folder):
            os.makedirs(folder, mode=0o755)
            if log:
                print(f"Created output directory: {folder}")
    # create log file
    logger = setup_logging(f'''{args.out}.cpheno.log'''.replace('//','/'))
    logger.info('Concat script for multiple phenotype files')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        args.files,
        args.out,
    ]
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("Concat phenotype table")
        logger.info("*"*60)
        logger.info(f"files:         {args.files}")
        logger.info(f"output prefix:  {args.out}")
        logger.info("*"*60 + "\n")
    return args,logger

t_start = time.time()
args,logger = main()
df = pd.concat([pd.read_csv(file,sep='\t',index_col=0,) for file in args.files],axis=1).reset_index()
df = df.groupby(df.columns[0]).mean()
dup_colloc = df.columns.duplicated()
if dup_colloc.sum()>0:
    logger.info(f'Duplicated phenotype: {set(df.columns[dup_colloc])}')
    logger.info('It will not generate .tsv file before changing these duplication.')
else:
    df.fillna('NA').to_csv(f'{args.out}.cpheno.tsv',sep='\t')
    print(f'Saved in {args.out}.cpheno.tsv')
lt = time.localtime()
endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
logger.info(endinfo)