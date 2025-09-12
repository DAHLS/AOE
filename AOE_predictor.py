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
