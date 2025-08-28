# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:50:42 2023

@author: asda
"""
import pandas as pd


curis_data = pd.read_excel('Spring/Data/2025-spring_curis_scopus_wos_combined.xlsx', sheet_name=0)
scwos_data = pd.read_excel('Spring/Data/2025-spring_curis_scopus_wos_combined.xlsx', sheet_name=3)
scwos_data['DOI'] = scwos_data['DOI'].str.lower()
curis_data['DOI'] = curis_data['DOI'].str.lower()

mask = scwos_data['DOI'].isin(curis_data['DOI'])
NotInCuris = scwos_data[~mask]

with pd.ExcelWriter("slutsoegning_2025_spring.xlsx") as writer:
    NotInCuris.to_excel(writer)