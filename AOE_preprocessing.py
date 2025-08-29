# -*- coding: utf-8 -*-
"""
Preprocessing of PURE report Org dump, as defined by Org_dump.json
Containing information about article affiliation
"""

import sys
import datetime
import pandas as pd
#import pickle
#from sklearn.feature_extraction.text import TfidfVectorizer

#'data/Org_dump.xlsx'
if len(sys.argv) < 2:
   print('Requires path to CURIS data')
   sys.exit()
data_path = sys.argv[1] 

pure_df = pd.read_excel(data_path)
pure_df.dropna(subset=['Orgs_parents'], inplace=True) 

#Reduce organization data to faculty affiliation
def extract_parent_org(row):
    split_string = row['Orgs_parents'].split('|')
    for element in split_string:
        if "Faculty" in element:
            return element.strip()
    return "Københavns Universitet"  # Return None if no element contains "Faculty"

pure_df['Orgs_parents'] = pure_df.apply(lambda row: extract_parent_org(row), axis=1)

replacements = {
    'Faculty Management': 'Faculty of Humanities',
    'Faculty Service': 'Københavns Universitet',
    'Faculty Services': 'Københavns Universitet',
    'Faculty of Pharmaceutical Sciences': 'Faculty of Health and Medical Sciences',
    'Faculty Services': 'Københavns Universitet',
    'Faculty of Life Sciences': 'Faculty of Science'
}

pure_df['Orgs_parents'] = pure_df['Orgs_parents'].replace(replacements)

#Combine textual data for modeling
pure_df['Text'] = pure_df[['Person', 'Title', 'Subtitle', 'Abs', 'Jn', 'Tihost']].apply(
    lambda row: ' '.join(row.dropna()), axis=1)

#Dump the processed pure data
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
pure_df.to_excel(f'data/{data_path.split('/')[-1].split('.')[0]}_processed_{timestamp}.xlsx', index=False)
