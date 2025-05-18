import pandas as pd
import argparse
# python mimic4noisy/subset_data.py --tasks phenotyping in-hospital-mortality
def create_subset(input_file, output_file, subset_ratio):
    """
    Create a subset of a CSV file and save it.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the subset CSV file.
        subset_ratio (float): Ratio of the data to include in the subset (0.0 to 1.0).
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)

        # Calculate the subset size
        subset_size = int(len(df) * subset_ratio)

        # Create the subset
        subset_df = df.sample(n=subset_size, random_state=42)

        # Save the subset to a new file
        subset_df.to_csv(output_file, index=False)

        print(f"Subset created successfully with ratio {subset_ratio}. Saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while creating a subset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate CSV files and create subsets.")
    parser.add_argument('--tasks', nargs='+', default=['diagnosis'],
                        help="Tasks to process.")
    parser.add_argument('--subset_ratio', type=float, default=0.05,
                        help="Ratio of data to include in the subset (0.0 to 1.0).")
    args = parser.parse_args()

    for task in args.tasks:
        # Concatenate train and validation files
        concatenated_file = f'data/{task}/train_val_multimodal_listfile.csv'

        # Create a subset of the concatenated file
        subset_file = f'data/{task}/train_val_multimodal_listfile_subset.csv'
        create_subset(concatenated_file, subset_file, args.subset_ratio)
        test_file = f'data/{task}/test_multimodal_listfile.csv'
        subset_test_file = f'data/{task}/test_multimodal_listfile_subset.csv'
        create_subset(concatenated_file, subset_test_file, args.subset_ratio)