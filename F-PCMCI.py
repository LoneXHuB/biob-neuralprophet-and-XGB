import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fpcmci.FPCMCI import FPCMCI
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from fpcmci.basics.constants import LabelType
from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel

# Function to compute causal relationships using FPCMCI
def compute_fpcmci_causality_with_heatmap(csv_file):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Drop non-numeric columns (FPCMCI requires numeric data)
    numeric_data = data.select_dtypes(include=[np.number])

    # Convert DataFrame to the required format for FPCMCI
    array_data = numeric_data.values
    df = Data(array_data)

    # Define the parameters for FPCMCI
    alpha = 0.05
    min_lag = 1
    max_lag = 5

    # Initialize FPCMCI
    fpcmci = FPCMCI(
        df,
        f_alpha=alpha,
        pcmci_alpha=alpha,
        min_lag=min_lag,
        max_lag=max_lag,
        sel_method=TE(TEestimator.Kraskov),  # Gaussian estimator for causal discovery
        val_condtest=GPDC(significance='analytic'),
        verbosity=CPLevel.INFO,  # Debugging verbosity
    )

    # Run FPCMCI
    sel_var, cm = fpcmci.run()

    # Save the DAG visualization
    fpcmci.dag(label_type=LabelType.Lag, node_layout='circular')
    dag_file = f"./data/results/{os.path.splitext(os.path.basename(csv_file))[0]}_dag.png"
    fpcmci.save_dag(dag_file)
    print(f"DAG visualization saved to '{dag_file}'")

    # Extract the causal matrix
    influence_matrix = cm.val_matrix[:, :, 1]
    p_value_matrix = cm.p_matrix[:, :, 1]

    # Create DataFrames for matrices
    influence_df = pd.DataFrame(influence_matrix, index=numeric_data.columns, columns=numeric_data.columns)
    p_value_df = pd.DataFrame(np.round(p_value_matrix, 2), index=numeric_data.columns, columns=numeric_data.columns)

    # Save the results as CSV
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    influence_csv_file = f"./data/results/{base_name}_influence_matrix.csv"
    p_value_csv_file = f"./data/results/{base_name}_p_value_matrix.csv"

    os.makedirs(os.path.dirname(influence_csv_file), exist_ok=True)

    influence_df.to_csv(influence_csv_file)
    p_value_df.to_csv(p_value_csv_file)

    print(f"Influence matrix saved to '{influence_csv_file}'")
    print(f"P-value matrix saved to '{p_value_csv_file}'")

    # Generate the heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        influence_df,
        annot=p_value_df.map(lambda x: f"{x:.2f}"),
        fmt='',  # Necessary for custom annotations
        cmap='coolwarm',
        cbar_kws={'label': 'Influence Score Intensity'},
        linewidths=0.5
    )

    plt.title("Influence Heatmap with P-values")
    plt.xlabel("Cause Variables")
    plt.ylabel("Effect Variables")

    # Save the heatmap as an image
    heatmap_file = f"./data/results/{base_name}_causal_matrix.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to '{heatmap_file}'")

# Example usage
data_file = "9705UE.csv"
compute_fpcmci_causality_with_heatmap(f"data/dec_4/{data_file}")
