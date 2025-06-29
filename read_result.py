import os
import csv

# Set your method and metric here
method = "midasv31"
metric = "disp_lstq"


TASKS = [
    ["Metric3Dv2", "metric"],
    ["Metric3Dv2", "dep_lstq"],
    
    ["MiDasV31", "disp_lstq"],
    ["MiDasV31", "dep_lstq"],

    ["DiffE2EFT", "dep_lstq"],
    ["GenPercept", "dep_lstq"],
]

for task in TASKS:
    METHOD_NAME, METRIC_NAME = task
    # Dataset order as required
    datasets = [
        "NYUv2", "KITTI", "ETH3D", "iBims-1", "GSO",
        "Sintel", "DDAD", "DIODE", "Spring", "HAMMER"
    ]

    base_dir = f"results/{METHOD_NAME.lower()}"

    print(f"{METHOD_NAME} - {METRIC_NAME}: Dataset\tAbs Rel\tDelta < 1.25")
    for dataset in datasets:
        metrics_path = os.path.join(
            base_dir, f"eval_{METHOD_NAME.lower()}-{METRIC_NAME}_{dataset}", "metrics.csv"
        )
        if not os.path.exists(metrics_path):
            print(f"{dataset}\tN/A\tN/A")
            continue
        with open(metrics_path, newline='') as csvfile:
            rows = list(csv.reader(csvfile))
            # Find the last row that starts with 'Average'
            row = rows[-1]
            # Print dataset, first two elements after 'Average'
            print(f"{dataset}\t{float(row[1])*100:.2f}\t{float(row[2])*100:.2f}")
