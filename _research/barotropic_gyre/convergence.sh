#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J barotropic_gyre
#SBATCH -A m4259
#SBATCH -t 02:00:00

# Replace this path with the path to the git repo
gitdir="${HOME}/.julia/dev/Moka.jl"
driver="src/driver/mpas_ocean.jl"
execut="${gitdir}/${driver}"

cd "${gitdir}/_research/barotropic_gyre"

for dir in 120km 60km 30km; do
    echo "Setting up $dir"
    mkdir -p $dir
    
    # Extract resolution integer
    res=${dir%km}

    # Generate initial mesh with python
    python generate_mesh.py --res $res --dir $dir
    
    # Copy config template and adjust dt if necessary based on resolution to maintain CFL
    # For 120km maybe dt=20m, 60km dt=10m, 30km dt=5m
    if [ "$res" = "120" ]; then
        dt="0000_00:20:00"
    elif [ "$res" = "60" ]; then
        dt="0000_00:10:00"
    elif [ "$res" = "30" ]; then
        dt="0000_00:05:00"
    else
        dt="0000_00:02:00"
    fi
    
    sed "s/config_dt: .*/config_dt: $dt/" config_template.yml > $dir/config.yml

    cd $dir

    start=$(date +%s.%N)
    
    echo "Running julia on $dir"
    julia -O0 --color=yes --project=$gitdir -- $execut config.yml

    end=$(date +%s.%N)
    
    runtime=$(awk -v start=$start -v end=$end 'BEGIN {print end - start}')
    echo "${dir} ran in ${runtime} secs"

    cd ..
done

# Finally, run compare.jl to post-process and plot
echo "Running compare.jl to generate plots and convergence stats"
echo "julia --project=$gitdir compare.jl"
