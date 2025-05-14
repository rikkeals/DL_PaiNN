#!/bin/bash

# === SETTINGS ===
project_dir="/zhome/4d/5/147570/02456_painn_project"   

# Define the sweep variables and their values
layers_list=(1 3)
features_list=(64 128 256)
num_rbf_features_list=(16 20 32)
cutoff_list=(4.0 5.0 6.0)
target_list=(0 2 7)


# === GRID SEARCH ===
for layers in "${layers_list[@]}"; do
  for features in "${features_list[@]}"; do
    for num_rbf_features in "${num_rbf_features_list[@]}"; do
      for cutoff in "${cutoff_list[@]}"; do
        for target in "${target_list[@]}"; do

          # Define the job name and script name
          job_name="PaiNN_layers${layers}_feat${features}_rbf${num_rbf_features}_cutoff${cutoff}_target${target}"
          script_name="train_layers${layers}_feat${features}_rbf${num_rbf_features}_cutoff${cutoff}_target${target}.sh"

          # Create the job script
          cat > $script_name <<EOL

# === JOB SCRIPT ===
#!/bin/sh
#BSUB -q gpua100
#BSUB -J $job_name
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -W 10:00
#BSUB -o ${job_name}.out
#BSUB -e ${job_name}.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate painn

cd $project_dir

python run_model.py --num_message_passing_layers $layers --num_features $features --num_rbf_features $num_rbf_features --cutoff_dist $cutoff --target $target

EOL

          # Submit the job
          bsub < $script_name

        done
      done
    done
  done
done
