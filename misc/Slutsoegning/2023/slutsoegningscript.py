# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:50:42 2023

@author: asda
"""
import pandas as pd


curis_data = pd.read_excel('data/curis_scopus_wos_combined.xlsx', sheet_name=0)
scwos_data = pd.read_excel('data/curis_scopus_wos_combined.xlsx', sheet_name=3)
scwos_data['DOI'] = scwos_data['DOI'].str.lower()
curis_data['DOI'] = curis_data['DOI'].str.lower()
NotInCuris = list()


for j, i in enumerate(scwos_data['DOI']):
    if i not in set(curis_data['DOI']):
        NotInCuris.append(scwos_data.loc[j])

NotInCuris_data = pd.DataFrame(NotInCuris)


with pd.ExcelWriter("slutsoegning_2023.xlsx") as writer:
    NotInCuris_data.to_excel(writer)