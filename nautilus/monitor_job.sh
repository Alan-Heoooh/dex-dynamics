#!/bin/bash

# Check input
if [ -z "$1" ]; then
  echo "Usage: $0 <job-name>"
  exit 1
fi

JOB_NAME="$1"

echo "Monitoring Job: $JOB_NAME"

# Wait for the Job's Pod to be created
echo "Waiting for pod to be created..."
while true; do
  POD_NAME=$(kubectl get pods --selector=job-name=$JOB_NAME -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)
  if [ -n "$POD_NAME" ]; then
    break
  fi
  sleep 1
done

echo "Found Pod: $POD_NAME"
echo "Showing Job status and streaming logs (Ctrl+C to exit)"

# Main loop: show job & pod status, then logs
while true; do
  echo
  echo "===== Job Status @ $(date) ====="
  kubectl get job $JOB_NAME
  kubectl describe job $JOB_NAME | grep -E "Active|Succeeded|Failed"

  echo
  echo "===== Pod Status ====="
  kubectl get pod $POD_NAME
  
  echo
  echo "===== Logs from $POD_NAME ====="
  kubectl logs -f $POD_NAME
  echo "===== End of Logs ====="

  sleep 2
done
