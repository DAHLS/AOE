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


#'Slutsoegning/2025/slutsogning_2025_spring_wip.xlsx'
if len(sys.argv) < 2:
    print("Please provide path to data for estimation.")
    sys.exit()
data_path = sys.argv[1]

new_df = pd.read_excel(data_path)
new_df['Text'] = new_df[['AU', 'TI', 'JN']].apply(
    lambda row: ' '.join(row.dropna()), axis=1)

#TODO add more flexiable model choice 
knn = joblib.load('models/AOE_kNN-model_20250912_190303.pkl')
vectorizer = joblib.load('models/AOE_tfidf-bow_20250912_190303.pkl')
X_new = vectorizer.transform(new_df['Text'])

predicted_labels = knn.predict(X_new)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
new_df['Predicted_Org_parents'] = predicted_labels
new_df.to_excel(f'output/{data_path.split('/')[-1].split('.')[0]}_labeled_{timestamp}.xlsx', index=False)

"""
import pandas as pd
import joblib
import datetime
import sys
import os
import glob

def select_file_from_directory(directory, file_extension, description):
    #Prompts the user to select a file from a given directory based on extension,
    #sorted by last modification time (newest first).
    #
    #Args:
    #    directory (str): The directory to search in.
    #    file_extension (str): The file extension to filter by (e.g., '.pkl').
    #    description (str): A description of the file type for the prompt.
    #
    #Returns:
    #    str: The full path to the selected file.
    pattern = os.path.join(directory, f"*{file_extension}")
    files = glob.glob(pattern)

    if not files:
        print(f"Error: No {description} files found in '{directory}' with extension '{file_extension}'.")
        sys.exit()

    # Sort files by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)

    print(f"\nAvailable {description} files in '{directory}' (sorted by modification time, newest first):")
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i + 1}. {filename} (Modified: {mod_time})")

    while True:
        try:
            choice = input(f"\nSelect the {description} number (1-{len(files)}): ")
            index = int(choice) - 1
            if 0 <= index < len(files):
                return files[index]
            else:
                print(f"Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            sys.exit()


# --- Main Script Execution ---

# 1. Check for input data path argument
if len(sys.argv) < 2:
    print("Please provide path to data for estimation.")
    sys.exit()
data_path = sys.argv[1]

# 2. Load the input data
new_df = pd.read_excel(data_path)
new_df['Text'] = new_df[['AU', 'TI', 'JN']].apply(
    lambda row: ' '.join(row.dropna().astype(str)), axis=1) # Ensure all parts are strings before joining

# 3. Prompt user to select the kNN model file
print("\n--- Selecting Model File ---")
model_path = select_file_from_directory('models', '.pkl', 'kNN Model')

# 4. Prompt user to select the TF-IDF vectorizer file
print("\n--- Selecting Vectorizer File ---")
vectorizer_path = select_file_from_directory('models', '.pkl', 'TF-IDF Vectorizer')

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
output_filename = f'output/{os.path.basename(data_path).split(".")[0]}_labeled_{timestamp}.xlsx'
new_df['Predicted_Org_parents'] = predicted_labels
new_df.to_excel(output_filename, index=False)

print(f"\nPredictions completed and saved to: {output_filename}")

"""
