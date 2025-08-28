#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 17:16:12 2025

@author: amk
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

new_df = pd.read_excel('Slutsoegning/2025/slutsogning_2025_spring_wip.xlsx')
new_df['Text'] = new_df[['AU', 'TI', 'JN']].apply(
    lambda row: ' '.join(row.dropna()), axis=1)

knn = joblib.load('orgafill_kNN-model.pkl')
vectorizer = joblib.load('orgaffil_tfidf-bow.pkl')
X_new = vectorizer.transform(new_df['Text'])

predicted_labels = knn.predict(X_new)
new_df['Predicted_Orgs_parents'] = predicted_labels
new_df.to_excel('labeled_new_data.xlsx', index=False)