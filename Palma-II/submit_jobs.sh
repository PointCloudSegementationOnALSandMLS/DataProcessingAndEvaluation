#!/bin/bash

# Set yourSBATCH options here
#SBATCH_OPTIONS="--partition=your_partition --time=your_time --mem=your_memory"

# Define your array of job commands
JOB_COMMANDS=(
    #"sbatch  --job-name=essen_mobile_kpconv --output=m_k_n_1.dat  train_essen_kpconv.sh ./configs/kpconv_essen_mobile_normalized.yml"
    #"sbatch  --job-name=essen_mobile_randlanet_normalized --output=a_r_n_1.dat train_essen_randlanet.sh ./configs/randlanet_essen_mobile_normalized.yml"
    #"sbatch  --job-name=essen_mobile_randlanet_ground --output=a_r_g_1.dat train_essen_randlanet.sh ./configs/randlanet_essen_mobile_ground.yml"
    #"sbatch  --job-name=essen_mobile_randlanet_hedge --output=a_r_h_1.dat train_essen_randlanet.sh ./configs/randlanet_essen_mobile_hedge.yml"
    #"sbatch  --job-name=essen_aerial_randlanet --output=a_r.dat train_essen_randlanet.sh ./configs/randlanet_essen_aerial.yml"
    #"sbatch  --job-name=essen_aerial_kpconv --output=a_k.dat  train_essen_kpconv.sh ./configs/kpconv_essen_aerial.yml"
    #"sbatch  --job-name=essen_aerial_randlanet_normalized --output=a_r_n.dat train_essen_randlanet.sh ./configs/randlanet_essen_aerial_normalized.yml"
    "sbatch  --job-name=essen_aerial_kpconv_normalized --output=a_k_n.dat  train_essen_kpconv.sh ./configs/kpconv_essen_aerial_normalized.yml"
    #"sbatch  --job-name=essen_aerial_randlanet_ground --output=a_r_g.dat train_essen_randlanet.sh ./configs/randlanet_essen_aerial_ground.yml"
    #"sbatch  --job-name=essen_aerial_randlanet_hedge --output=a_r_h.dat train_essen_randlanet.sh ./configs/randlanet_essen_aerial_hedge.yml"
    #"sbatch  --job-name=essen_aerial_randlanet_gorund_hedge --output=a_r_gh.dat train_essen_randlanet.sh ./configs/randlanet_essen_aerial_ground_hedge.yml"
    #"sbatch  --job-name=essen_aerial_kpconv_tf --output=aerial_kpconv_tf.dat train_essen_kpconv_tf.sh ./configs/kpconv_essen_aerial_normalized_tf.yml"
    #"sbatch  --job-name=essen_mobile_kpconv_tf --output=mobile_kpconv_tf.dat train_essen_kpconv_tf.sh ./configs/kpconv_essen_mobile_tf.yml"
    #"sbatch  --job-name=essen_aerial_randlanet_ground_vertical --output=a_r_g_v.dat train_essen_randlanet.sh ./configs/randlanet_essen_aerial_vertical.yml"
    "sbatch  --job-name=essen_aerial_randlanet_ground_vertical --output=a_r_v.dat train_essen_randlanet.sh ./configs/randlanet_essen_aerial_vertical.yml"
    "sbatch  --job-name=essen_mobile_randlanet_ground_vertical --output=m_r_v.dat train_essen_randlanet.sh ./configs/randlanet_essen_mobile_vertical.yml"
# Add more job commands as needed
)

# Loop through the array and submit jobs
for cmd in "${JOB_COMMANDS[@]}"; do
    eval "$cmd"
    echo "Submitting job: $cmd"
done
