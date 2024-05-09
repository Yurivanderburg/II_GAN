#!/bin/bash
#SBATCH --nodes=3      
#SBATCH --ntasks=28
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --qos=6node_qos
#SBATCH --output=sh/gan_2.%J.out
#SBATCH --error=sh/gan_2.%J.err
    
#source ~/.bashrc
module swap intel gcc/9.2.0
module load pmix/2.2.2
source /etc/profile.d/conda.sh
conda activate gan
python3 gan_def_test.py
