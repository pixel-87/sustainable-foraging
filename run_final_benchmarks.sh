#!/usr/bin/env bash
# run_final_benchmarks.sh
# Executes the final 3-seed benchmark runs
# Runs seeds IN PARALLEL to fully utilize the 6-core/12-thread Ryzen CPU!

# AMD GPU Prefix for RX 6650 XT
ROCM_PREFIX="HSA_OVERRIDE_GFX_VERSION=10.3.0"
# Use uv run python and set num-cpus 4 for multiprocessing
TRAIN_CMD="uv run python scripts/train.py --preset fair --num-envs 4 --num-cpus 4"

echo "========================================================="
echo " Starting Sustainable Foraging Final Benchmarks"
echo " GPU: AMD RX 6650 XT (ROCm 10.3.0 Override)"
echo " CPU: Running 3 seeds in parallel (12 env threads total)"
echo "========================================================="

# Helper function to run an algorithm across 3 seeds IN PARALLEL
run_benchmark_parallel() {
    local lib=$1
    local algo=$2
    local timesteps=$3
    local extra_args=$4

    local total_target=$((timesteps * 3))

    echo ""
    echo ">>> Starting $algo ($lib) - $timesteps timesteps per seed (${total_target} total)"
    
    # Store process IDs to monitor them
    pids=""
    
    # Launch all 3 seeds in the background
    for seed in 1 2 3; do
        local run_name="${algo}_final_seed${seed}"
        local log_dir="logs/${run_name}"
        
        # Ensure log dir exists so we can redirect output
        mkdir -p "$log_dir"
        
        echo "  -> Launching Seed $seed (Logging stdout to $log_dir/stdout.log)..."
        
        # Execute the run in the background using '&'. 
        # We redirect stdout and stderr so it doesn't mess up our beautiful progress bar!
        env $ROCM_PREFIX uv run python scripts/train.py --preset fair --num-envs 4 --num-cpus 4 --library $lib --algorithm $algo --timesteps $timesteps --seed $seed --name $run_name $extra_args > "$log_dir/stdout.log" 2>&1 &
        pids="$pids $!"
    done
    
    echo "  -> Monitor started. Please wait..."
    
    local last_total_steps=0
    local last_time=$(date +%s)
    
    # Progress Monitor Loop
    while true; do
        # Sleep for a short interval so we get responsive updates
        sleep 10
        
        # Check if any of the 3 background PIDs are still running
        local any_running=false
        for pid in $pids; do
            if kill -0 $pid 2>/dev/null; then
                any_running=true
                break
            fi
        done
        
        # Calculate current total timesteps by reading the CSV files
        local current_total_steps=0
        for seed in 1 2 3; do
            local run_name="${algo}_final_seed${seed}"
            local csv_file="logs/${run_name}/metrics.csv"
            
            if [ -f "$csv_file" ]; then
                # Get the last line, split by comma, get column 2 (timestep)
                local last_ts=$(tail -n 1 "$csv_file" 2>/dev/null | cut -d',' -f2)
                # Ensure it's a number (ignores the CSV header)
                if [[ "$last_ts" =~ ^[0-9]+$ ]]; then
                    current_total_steps=$((current_total_steps + last_ts))
                fi
            fi
        done
        
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_time))
        
        if [ $time_diff -gt 0 ]; then
            local step_diff=$((current_total_steps - last_total_steps))
            local speed=$((step_diff / time_diff))
            
            # Format to "Millions" (e.g. 1.50M)
            local curr_m=$(awk -v steps="$current_total_steps" 'BEGIN { printf "%.2f", steps/1000000 }')
            local targ_m=$(awk -v steps="$total_target" 'BEGIN { printf "%.2f", steps/1000000 }')
            
            if [ $speed -gt 0 ]; then
                local remaining_steps=$((total_target - current_total_steps))
                if [ $remaining_steps -lt 0 ]; then remaining_steps=0; fi
                
                local eta_secs=$((remaining_steps / speed))
                local eta_hrs=$((eta_secs / 3600))
                local eta_mins_rem=$(((eta_secs % 3600) / 60))
                
                # The \r at the start overwrites the current line in the terminal!
                printf "\r    [Progress] %s: %sM / %sM steps | Speed: %d steps/s | ETA: %dh %dm    " "$algo" "$curr_m" "$targ_m" "$speed" "$eta_hrs" "$eta_mins_rem"
            else
                printf "\r    [Progress] %s: %sM / %sM steps | Speed: Calculating...                   " "$algo" "$curr_m" "$targ_m"
            fi
            
            last_total_steps=$current_total_steps
            last_time=$current_time
        fi
        
        # If all 3 finished, break the monitor loop
        if [ "$any_running" = false ]; then
            break
        fi
    done
    
    echo "" # Print a newline so the next print doesn't overwrite our final progress bar
    echo ">>> Finished $algo"
}

# ---------------------------------------------------------
# OFF-POLICY ALGORITHMS (3 Million Timesteps - Updated)
# ---------------------------------------------------------
OFF_POLICY_STEPS=3000000

run_benchmark_parallel "cleanrl" "dqn" $OFF_POLICY_STEPS "--lr 0.0003 --exploration-fraction 0.5 --target-network-frequency 200 --tau 1.0"
run_benchmark_parallel "cleanrl" "vdn" $OFF_POLICY_STEPS "--lr 0.0001 --exploration-fraction 0.5 --target-network-frequency 1 --tau 0.01"
run_benchmark_parallel "cleanrl" "qmix" $OFF_POLICY_STEPS "--lr 0.0001 --exploration-fraction 0.5 --target-network-frequency 1 --tau 0.01"

# ---------------------------------------------------------
# ON-POLICY ALGORITHMS (10 Million Timesteps - Scaled Lower Bound)
# Hyperparameters: Entropy Coef = 0.001
# ---------------------------------------------------------
ON_POLICY_STEPS=10000000

# run_benchmark_parallel "sb3" "ppo" $ON_POLICY_STEPS "--lr 0.0003 --ent-coef 0.001"
# run_benchmark_parallel "cleanrl" "mappo" $ON_POLICY_STEPS "--lr 0.0003 --ent-coef 0.001"
# run_benchmark_parallel "sb3" "a2c" $ON_POLICY_STEPS "--lr 0.0005 --ent-coef 0.001"

echo ""
echo "========================================================="
echo " All Benchmarks Completed Successfully!"
echo "========================================================="