#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:49:15 2025

@author: amk
"""

import sys
import pandas as pd
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

#'data/Org_dump_processed.xlsx'
if len(sys.argv) < 2:
    "Provide path to preprocessed data"
    sys.exit()
data_path = sys.argv[1]

pure_df = pd.read_excel(data_path)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(pure_df['Text'])

y = pure_df['Orgs_parents'].values
y_stem = pure_df['Area'].values
X = tfidf_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=42)
classifier_type = 'kNN'
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Training set size: " + str(X_train.shape))
print("Test set size: " + str(X_test.shape))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#stem eller ej
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_stem,
                                                            stratify=y_stem,
                                                            test_size=0.2,
                                                            random_state=42)
classifier_s = KNeighborsClassifier(n_neighbors=5)
classifier_s.fit(X_train_s, y_train_s)
y_pred_s = classifier_s.predict(X_test_s)

print("Training set size: " + str(X_train_s.shape))
print("Test set size: " + str(X_test_s.shape))
print(f"Accuracy: {accuracy_score(y_test_s, y_pred_s):.2f}")
print("\nClassification Report:")
print(classification_report(y_test_s, y_pred_s))


#Save trained model and tfidf vectorization
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(vectorizer, f'models/AOE_tfidf-bow_{timestamp}.pkl')
joblib.dump(classifier, f'models/AOE_{classifier_type}-model_{timestamp}.pkl')
joblib.dump(classifier_s, f'models/AOE_{classifier_type}-stem-model_{timestamp}.pkl')