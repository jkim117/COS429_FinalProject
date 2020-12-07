#!/bin/bash                                                                                                    

#SBATCH --job-name=COMB_HALF-test                                                                            
#SBATCH --nodes=1                                                                                              
#SBATCH --ntasks=1                                                                                             
#SBATCH --gres=gpu:1                                                                                           
#SBATCH --cpus-per-task=1                                                                                      
#SBATCH --time=24:00:00                                                                                        
#SBATCH --mail-type=begin                                                                                      
#SBATCH --mail-type=end                                                                                        
#SBATCH --mail-type=fail                                                                                       
#SBATCH --mail-user=minaba@princeton.edu                                                                       

module load anaconda3
conda activate cos429
python inpaint_combo.py