# -*- coding: utf-8 -*-
from genericpath import isfile
import os
import argparse
import time
import socket
from ._common.log import setup_logging
from rust2py.g2phy import geno2phy
from ext.iqtree import main as iqtree  # Ensure iqtree dependencies are checked at startup
def main(log:bool=True):
    t_start = time.time()
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__)
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('-vcf','--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('-bfile','--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    geno_group.add_argument('-npy','--npy', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .npz, .snp, .idv)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-t', '--threads', type=int, default=1,
                               help='Specify number of threads used'
                                   '(default: %(default)s)')
    optional_group.add_argument('-o', '--out', type=str, default=None,
                               help='Output directory for results'
                                   '(default: %(default)s)')
    optional_group.add_argument('-prefix','--prefix',type=str,default=None,
                               help='prefix of output file'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    # Determine genotype file
    if args.vcf:
        gfile = args.vcf
        args.prefix = os.path.basename(gfile).replace('.gz','').replace('.vcf','') if args.prefix is None else args.prefix
    elif args.bfile:
        gfile = args.bfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    elif args.npy:
        gfile = args.npy
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    gfile = gfile.replace('\\','/')
    args.out = os.path.dirname(gfile) if args.out is None else args.out
    os.makedirs(args.out,0o755,True) # create log file
    logger = setup_logging(f'''{args.out}/{args.prefix}.pca.log'''.replace('\\','/').replace('//','/'))
    logger.info('Principle Component Analysis Module')
    logger.info(f'Host: {socket.gethostname()}\n')
    # Print configuration summary
    if log:
        logger.info("*"*60)
        logger.info("PCA CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file: {gfile}")
        logger.info(f"Threads:       {args.threads}")
        logger.info(f"Output prefix: {args.out}/{args.prefix}")
        logger.info("*"*60 + "\n")
    
    geno2phy(genofile=gfile.replace('.gz','').replace('.vcf',''),
              filetype='vcf' if args.vcf else ('bfile' if args.bfile else 'npy'),
              out=args.out,
              outprefix=args.prefix,
              write_phylip=True,
              write_fasta=False,
              write_nexus=False,
              write_nexus_binary=False,used_sites=False)
    os.system(f'{iqtree()} -s {args.out}/{args.prefix}.phy -st DNA -m MFP+ASC -bb 1000 -alrt 1000 -nt AUTO')
    if os.path.isfile(f'{args.out}/{args.prefix}.phy.varsites.phy'):
        os.remove(f'{args.out}/{args.prefix}.phy')
        os.system(f'{iqtree()} -s {args.out}/{args.prefix}.phy.varsites.phy -pre {args.out}/{args.prefix} -st DNA -m MFP+ASC -bb 1000 -alrt 1000 -nt AUTO -T {args.threads}')
    
    # Finish logging
    lt = time.localtime()
    endinfo = f'\nFinished, total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    logger.info(endinfo)
