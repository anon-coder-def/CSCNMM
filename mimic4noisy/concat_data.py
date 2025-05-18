import pandas as pd
import argparse

def concatenate_csv(file1, file2, output_file, task, axis=0):
    """
    Concatenate two CSV files and save the result.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        output_file (str): Path to save the concatenated CSV file.
        axis (int): Axis to concatenate along. Default is 0 (row-wise concatenation).
    """
    try:
        # Load the CSV files into DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Concatenate the DataFrames
        concatenated_df = pd.concat([df1, df2], axis=axis)

        # Save the concatenated DataFrame to a new file
        concatenated_df.to_csv(output_file, index=False)

        print(f"Files concatenated successfully. Saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while concatenating CSV files: {e}")

# Example usage



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add label noise to specified tasks.")
    parser.add_argument('--tasks', nargs='+', default=['diagnosis'],
                        help="Tasks to add noise to.")
    args = parser.parse_args()

    task = args.tasks
    
    for task in args.tasks:
        file1 = f'data/{task}/train_multimodal_listfile.csv'
        file2 = f'data/{task}/val_multimodal_listfile.csv'
        output_file = f'data/{task}/train_val_multimodal_listfile.csv'
        concatenate_csv(file1, file2, output_file, task)