import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tigramite.data_processing import DataFrame as TigraDataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


# Function to compute causal relationships using PCMCI
def compute_pcmci_causality_with_heatmap(csv_file):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Drop non-numeric columns (PCMCI requires numeric data)
    numeric_data = data.select_dtypes(include=[np.number])

    # Wrap into Tigramite's DataFrame (NOT pandas.DataFrame)
    tigra_df = TigraDataFrame(data=numeric_data.values,
                              var_names=list(numeric_data.columns))

    # Define the independence tests
    parcorr = ParCorr(significance='analytic')

    # Initialize PCMCI with the non-linear test
    pcmci = PCMCI(dataframe=tigra_df, cond_ind_test=parcorr)
    # Or use parcorr instead:
    # pcmci = PCMCI(dataframe=tigra_df, cond_ind_test=parcorr)

    print("Running pcmci....")
    # Run PCMCI
    results = pcmci.run_pcmci(tau_min=0, tau_max=5, pc_alpha=0.05)
    print("Done.")
    # Extract full p-value and influence score matrices for lag 0
    influence_matrix = results['val_matrix'][:, :, 0]
    p_value_matrix = results['p_matrix'][:, :, 0]

    # Create DataFrames for the matrices
    influence_df = pd.DataFrame(influence_matrix,
                                index=numeric_data.columns,
                                columns=numeric_data.columns)
    p_value_df = pd.DataFrame(np.round(p_value_matrix, 2),
                              index=numeric_data.columns,
                              columns=numeric_data.columns)

    # Save the results as CSV
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    results_dir = './data/results'
    os.makedirs(results_dir, exist_ok=True)

    influence_csv_file = os.path.join(results_dir, f'{base_name}_influence_matrix_latest.csv')
    p_value_csv_file = os.path.join(results_dir, f'{base_name}_p_value_matrix_latest.csv')

    influence_df.to_csv(influence_csv_file)
    p_value_df.to_csv(p_value_csv_file)

    print(f"Influence matrix saved to '{influence_csv_file}'")
    print(f"P-value matrix saved to '{p_value_csv_file}'")

    # Generate the heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        influence_df,
        annot=p_value_df.astype(str),  # show p-values as annotations
        fmt='',
        cmap='coolwarm',
        cbar_kws={'label': 'Influence Score Intensity'},
        linewidths=0.5
    )

    plt.title('Influence Heatmap with P-values')
    plt.xlabel('Cause Variables')
    plt.ylabel('Effect Variables')

    # Save the heatmap as an image
    heatmap_file = os.path.join(results_dir, f'{base_name}_causal_matrix_latest.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to '{heatmap_file}'")


# Example usage
data_file = "9610PBUE.csv"
compute_pcmci_causality_with_heatmap(f'data/dec_4/{data_file}')