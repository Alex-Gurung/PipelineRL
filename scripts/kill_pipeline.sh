#!/usr/bin/env bash
set -euo pipefail

# Kill PipelineRL-related processes hard, including parents of defunct zombies.
# Patterns cover vLLM engines, actor, preprocessor, finetune (accelerate),
# W&B helpers, and vLLM EngineCore workers.

declare -a MATCH_PATTERNS=(
  "pipelinerl.entrypoints.run_vllm"
  "pipelinerl/entrypoints/run_vllm"
  "pipelinerl.entrypoints.run_actor"
  "pipelinerl/entrypoints/run_actor"
  "pipelinerl.entrypoints.run_preprocess"
  "pipelinerl/entrypoints/run_preprocess"
  "pipelinerl/entrypoints/run_finetune.py"
  "accelerate.commands.launch"
  "VLLM::EngineCore"
  "wandb-core"
  "gpu_stats"
  "resource_tracker.*main;main\(53\)"
)

SELF_PID=$$

mapfile -t PS_LINES < <(ps -eo pid=,ppid=,stat=,command= -ww)

declare -A TO_KILL
declare -A PARENTS_TO_KILL

for line in "${PS_LINES[@]}"; do
  pid=$(awk '{print $1}' <<<"$line")
  ppid=$(awk '{print $2}' <<<"$line")
  stat=$(awk '{print $3}' <<<"$line")
  cmd=${line#*$stat }  # command (after stat)

  # Skip our own shell and init
  [[ "$pid" == "$SELF_PID" || "$pid" == "1" ]] && continue

  for pat in "${MATCH_PATTERNS[@]}"; do
    if [[ "$cmd" =~ $pat ]]; then
      if [[ "$stat" == *Z* ]]; then
        # Zombie; kill parent instead if possible
        if [[ "$ppid" != "1" ]]; then
          PARENTS_TO_KILL[$ppid]=1
        fi
      else
        TO_KILL[$pid]=1
      fi
      break
    fi
  done
done

kill_list=()
for k in "${!TO_KILL[@]}"; do kill_list+=("$k"); done
for k in "${!PARENTS_TO_KILL[@]}"; do kill_list+=("$k"); done

if ((${#kill_list[@]}==0)); then
  echo "No matching processes found to kill."
  exit 0
fi

echo "Killing PIDs with SIGKILL: ${kill_list[*]}"
kill -9 "${kill_list[@]}" 2>/dev/null || true

# Double check lingering zombies: list again for visibility
echo "Remaining matching processes (if any):"
ps -eo pid=,ppid=,stat=,command= -ww | \ 
  awk 'BEGIN{IGNORECASE=1} /pipelinerl\.|VLLM::EngineCore|wandb-core|gpu_stats|accelerate\.commands\.launch|resource_tracker/ {print}' || true

