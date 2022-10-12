#!/usr/bin/env python

import os

#def mkdir_p(dir):
#    '''make a directory (dir) if it doesn't exist'''
#    if not os.path.exists(dir):
#        os.mkdir(dir)
    

job_directory = os.getcwd()

#data_dir = os.path.join(job_directory, '/project')

# Make top level directories
#mkdir_p(job_directory)
#mkdir_p(data_dir)

lizards=["LizardA","LizardB"]

for lizard in lizards:

    job_file = os.path.join(job_directory,"%s.job" %lizard)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % lizard)
        fh.writelines("#SBATCH --output=.out/%s.out\n" % lizard)
        fh.writelines("#SBATCH --error=.out/%s.err\n" % lizard)
        fh.writelines("#SBATCH --time=2:00:00\n")
        fh.writelines("#SBATCH --mem=1G\n")
        fh.writelines("#SBATCH --partition=compute\n")
        fh.writelines("#SBATCH --account=research-ceg-gse\n")
        fh.writelines("python teste_gpu.py $SLURM_ARRAY_TASK_ID $SLURM_SUBMIT_DIR\n")

    os.system("sbatch %s" %job_file)

    
