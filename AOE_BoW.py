#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:49:15 2025

@author: amk
"""

import pandas as pd
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

pure_df = pd.read_excel('Org_dump_processed.xlsx')
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(pure_df['Text'])

y = pure_df['Orgs_parents'].values
X = tfidf_matrix
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#Save trained model and tfidf vectorization
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(vectorizer, f'models/AOE_tfidf-bow_{timestamp}.pkl')
joblib.dump(knn, f'models/AOE_kNN-model_{timestamp}.pkl')

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))