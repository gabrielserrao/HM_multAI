#!/bin/bash 

#SBATCH --partition=compute
#SBATCH --job-name=esmda_ 
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --account=research-ceg-gse
##SBATCH -o %a_pix2pix_log.txt
##SBATCH -e %a_pix2pix_err.txt
#SBATCH -o esmda_log.txt
#SBATCH -e esmda_err.txt

module load 2022r2


python ES-MDA_DARTS_v2.py 

