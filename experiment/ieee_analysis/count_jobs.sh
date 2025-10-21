#!/bin/bash

echo "=== IBM SHERBROOKE DATA ==="
echo ""
echo "Root files:"
for file in "2_experiment_results_target_depth.csv" "3_experiment_results.csv"; do
    if [ -f "$file" ]; then
        count=$(($(wc -l < "$file") - 1))
        echo "  $file: $count jobs"
    fi
done

echo ""
echo "IBM directory files:"
ibm_total=0
for file in ibm/*.csv; do
    if [ -f "$file" ]; then
        count=$(($(wc -l < "$file") - 1))
        echo "  $(basename $file): $count jobs"
        ibm_total=$((ibm_total + count))
    fi
done

# Count root files
root_2=$(($(wc -l < "2_experiment_results_target_depth.csv") - 1))
root_3=$(($(wc -l < "3_experiment_results.csv") - 1))
ibm_grand_total=$((root_2 + root_3 + ibm_total))

echo ""
echo "IBM Subtotals:"
echo "  Root files: $((root_2 + root_3)) jobs"
echo "  IBM directory: $ibm_total jobs"
echo "  TOTAL IBM: $ibm_grand_total jobs"
echo "  TOTAL SHOTS (10 per job): $((ibm_grand_total * 10))"

echo ""
echo "=== RIGETTI ANKAA-3 DATA ==="
echo ""
echo "Root file:"
rigetti_root_file="../qbraid/experiment_results_target_depth_20250903_221822_updated.csv"
if [ -f "$rigetti_root_file" ]; then
    count=$(($(wc -l < "$rigetti_root_file") - 1))
    echo "  experiment_results_target_depth_20250903_221822_updated.csv: $count jobs"
fi

echo ""
echo "AWS directory files:"
aws_total=0
for file in aws/*.csv; do
    if [ -f "$file" ]; then
        count=$(($(wc -l < "$file") - 1))
        echo "  $(basename $file): $count jobs"
        aws_total=$((aws_total + count))
    fi
done

rigetti_root=$(($(wc -l < "$rigetti_root_file") - 1))
rigetti_grand_total=$((rigetti_root + aws_total))

echo ""
echo "Rigetti Subtotals:"
echo "  Root file (qBraid): $rigetti_root jobs"
echo "  AWS directory: $aws_total jobs"
echo "  TOTAL RIGETTI: $rigetti_grand_total jobs"
echo "  TOTAL SHOTS (10 per job): $((rigetti_grand_total * 10))"

echo ""
echo "=== GRAND TOTAL ==="
total_jobs=$((ibm_grand_total + rigetti_grand_total))
total_shots=$((total_jobs * 10))
echo "Total Jobs: $total_jobs"
echo "Total Shots: $total_shots"
