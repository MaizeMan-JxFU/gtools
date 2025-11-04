from pyBLUP import mlm
from gfreader import breader,vcfreader
import pandas as pd
import numpy as np
import argparse
import time
import socket
import logging
import sys
import os
doc = '''
Examples:
  # Basic usage with VCF file
  --vcf genotypes.vcf --pheno phenotypes.txt --out results
  
  # Using PLINK binary files with custom parameters
  --bfile genotypes --pheno phenotypes.txt --out results --grm gemma1 --qcov 5 --thread 8
  
  # Using external kinship matrix and disabling HighAC
  --vcf genotypes.vcf --pheno phenotypes.txt --out results --grm kinship_matrix.txt --qcov 10 --no-AC
  
  # Maximum performance with all threads
  --bfile genotypes --pheno phenotypes.txt --out results --grm VanRanden --qcov 3 --thread -1

File Formats:
  VCF/BFILE:    Standard VCF or PLINK binary format (bed/bim/fam)
  PHENO:        Tab-delimited file with sample IDs in first column and phenotypes in subsequent columns
  GRM File:     Space/tab-delimited kinship matrix file
  QCOV File:    Space/tab-delimited covariate matrix file
        
Citation:
  https://github.com/MaizeMan-JxFU/pyBLUP/
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docfile = os.path.join(script_dir,'../doc','demo.txt')
    doc = open(docfile, 'r',).read()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    required_group.add_argument('--pheno', type=str, required=True,
                               help='Phenotype file (tab-delimited with sample IDs in first column)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('--out', type=str, default='test',
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('--grm', type=str,
                               default='VanRanden',
                               help='Kinship matrix calculation method or path to pre-calculated GRM file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--qcov', type=str, default='3',
                               help='Number of principal components for Q matrix or path to covariate matrix file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--cov', type=str, default=None,
                               help='Path to Covariance file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--thread', type=int, default=-1,
                               help='Number of CPU threads to use (-1 for all available cores, default: %(default)s)')
    optional_group.add_argument('--AC', action='store_true', default=True,
                               help='Enable HighAC mode for GWAS (default: %(default)s)')
    optional_group.add_argument('--no-AC', action='store_false', dest='AC',
                               help='Disable HighAC mode')
    args = parser.parse_args()
    # Determine genotype file
    gfile = args.vcf if args.vcf else args.bfile
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        gfile,
        args.pheno,
        args.out,
        args.grm,
        args.qcov,
        args.cov,
        str(args.AC)
    ]
    # create log file
    if not os.path.exists(args.out):
        os.mkdir(args.out,0o755)
    
    filename = os.path.basename(gfile)
    logger = setup_logging(f'''{args.out}/{filename.replace('.vcf','').replace('.gz','')}.log'''.replace('//','/'))
    logger.info('High Performance Linear Mixed Model Solver for Genome-Wide Association Studies')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("GWAS LMM SOLVER CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file:    {gfile}")
        logger.info(f"Phenotype file:   {args.pheno}")
        logger.info(f"Output directory: {args.out}")
        logger.info(f"GRM method:       {args.grm}")
        logger.info(f"Q matrix:         {args.qcov}")
        logger.info(f"Covariant matrix: {args.cov}")
        logger.info(f"Threads:          {args.thread} ({'All cores' if args.thread == -1 else 'User specified'})")
        logger.info(f"HighAC mode:      {args.AC}")
        logger.info("*"*60 + "\n")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.out):
        os.makedirs(args.out, mode=0o755)
        if log:
            print(f"Created output directory: {args.out}")
    return gfile,args,logger

t_start = time.time()
gfile,args,logger = main()