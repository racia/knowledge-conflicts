#!/usr/bin/env bash

for i in {1..10}; do
    jobid=$(sbatch -p dev_gpu_h100,dev_gpu_a100_il scripts/plant_conflicts_Raziye.sh | awk '{print $4}')
    echo "Submitted job $jobid"

    # Wait until job disappears from squeue
    while squeue -j "$jobid" >/dev/null 2>&1; do
        sleep 5
    done

    echo "Job $jobid finished."
done
