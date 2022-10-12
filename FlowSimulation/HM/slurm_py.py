#!/usr/bin/env python

import os

def mkdir_p(dir):
   '''make a directory (dir) if it doesn't exist'''
   if not os.path.exists(dir):
       os.mkdir(dir)
    

job_directory = os.getcwd()

data_dir = os.path.join(job_directory, '/project')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)

lizards=["LizardA","LizardB"]

for lizard in lizards:

    job_file = os.path.join(job_directory,"%s.job" %lizard)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % lizard)
        fh.writelines("#SBATCH --output=%s.out\n" % lizard)
        fh.writelines("#SBATCH --error=%s.err\n" % lizard)
        fh.writelines("#SBATCH --time=2:00:00\n")
        fh.writelines("#SBATCH --mem=1G\n")
        fh.writelines("#SBATCH --partition=compute\n")
        fh.writelines("#SBATCH --account=research-ceg-gse\n")
        fh.writelines(f'a={str(lizard)}\n')
        ##fh.writelines(f'b={str(destDir)}\n')
        ##fh.writelines(f'c={str(time_range[-1])}\n')
        fh.writelines(f'python teste_gpu.py $a\n')
        ##fh.writelines("python teste_gpu.py $SLURM_SUBMIT_DIR\n")

    os.system("sbatch %s" %job_file)

    
