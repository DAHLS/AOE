#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 17:16:12 2025

@author: amk
"""

import pandas as pd
import joblib
import datetime
import sys
import os
import re
import glob
import argparse
from pathlib import Path


def select_file_from_directory(directory, file_extension, description,
                               include_pattern=None, exclude_pattern=None):
    """
    Prompts user to select a file from a directory based on extension and optional filename patterns.
    
    Args:
        directory (str): Directory to search in.
        file_extension (str): File extension to filter by (e.g., '.pkl').
        description (str): Description for display/user prompt.
        include_pattern (str, optional): If provided, only files *containing* this substring
                                         (case/hyphen/underscore insensitive) in their filename are shown.
                                         E.g., 'tfidf' matches 'tf-idf', 'T_F_IDF', etc.
        exclude_pattern (str, optional): If provided, files *containing* this substring
                                         (same flexible matching) are excluded.

    Returns:
        str: Full path to selected file.
    """

    # Helper: normalize string by removing non-alphanumeric chars and lowercasing
    def normalize(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

    pattern = os.path.join(directory, f"*{file_extension}")
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"Error: No {description} files found in '{directory}' with extension '{file_extension}'.")
        sys.exit()

    # Pre-normalize patterns for comparison
    include_norm = normalize(include_pattern) if include_pattern else None
    exclude_norm = normalize(exclude_pattern) if exclude_pattern else None

    filtered_files = []
    for f in all_files:
        filename = os.path.basename(f)
        norm_filename = normalize(filename)

        # Check inclusion/exclusion using normalized forms
        if include_norm and include_norm not in norm_filename:
            continue
        if exclude_norm and exclude_norm in norm_filename:
            continue
        filtered_files.append(f)

    if not filtered_files:
        incl_desc = f'containing "{include_pattern}" ' if include_pattern else ""
        excl_desc = f'not containing "{exclude_pattern}" ' if exclude_pattern else ""
        print(f"Error: No {description} files found in '{directory}' matching "
              f"{incl_desc}{excl_desc}(extension: '{file_extension}').")
        sys.exit()

    # Sort by modification time (newest first)
    filtered_files.sort(key=os.path.getmtime, reverse=True)
    print(f"\nAvailable {description} files in '{directory}' "
          f"(sorted by modification time, newest first):")
    for i, file_path in enumerate(filtered_files):
        filename = os.path.basename(file_path)
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i + 1}. {filename} (Modified: {mod_time})")
    while True:
        try:
            choice = input(f"\nSelect the {description} number (1-{len(filtered_files)}): ")
            index = int(choice) - 1
            if 0 <= index < len(filtered_files):
                return filtered_files[index]
            else:
                print(f"Please enter a number between 1 and {len(filtered_files)}.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            sys.exit()
            

# --- Main Script Execution ---

def main():
    parser = argparse.ArgumentParser(
        description='Predict organizational affiliation for publications')
    parser.add_argument('data_path', help='Path to data for estimation')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file (default: interactive selection)')
    parser.add_argument('--vectorizer', '-v', type=str, default=None,
                        help='Path to vectorizer file (default: interactive selection)')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                        help='Output directory (default: output/)')
    args = parser.parse_args()

    # 1. Check for input data path argument
    data_path = args.data_path

    # 2. Load the input data
    new_df = pd.read_excel(data_path)
    new_df['Text'] = new_df[['AU', 'TI', 'JN']].apply(
        lambda row: ' '.join(row.dropna().astype(str)), axis=1) # Ensure all parts are strings before joining

    # 3. Prompt user to select the model file
    print("\n--- Selecting Model File ---")
    if args.model:
        model_path = args.model
        print(f"Using specified model: {model_path}")
    else:
        model_path = select_file_from_directory(
            'models', '.pkl', 'kNN Model',
            exclude_pattern='tfidf')

    # 4. Prompt user to select the TF-IDF vectorizer file
    print("\n--- Selecting Vectorizer File ---")
    if args.vectorizer:
        vectorizer_path = args.vectorizer
        print(f"Using specified vectorizer: {vectorizer_path}")
    else:
        vectorizer_path = select_file_from_directory(
            'models', '.pkl', 'TF-IDF Vectorizer',
            include_pattern='tfidf')

    # 5. Load the selected model and vectorizer
    print(f"\nLoading model from: {model_path}")
    knn = joblib.load(model_path)
    print(f"Loading vectorizer from: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    # 6. Perform the prediction
    X_new = vectorizer.transform(new_df['Text'])
    predicted_labels = knn.predict(X_new)

    # 7. Save the results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f'{Path(data_path).stem}_labeled_{timestamp}.xlsx'
    new_df['Predicted_Org_parents'] = predicted_labels
    new_df.to_excel(output_filename, index=False)

    print(f"\nPredictions completed and saved to: {output_filename}")


if __name__ == "__main__":
    main()
