# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:12:00 2023

@author: asda
"""

import pandas as pd

scopus = pd.read_excel('SCOPUS_SLUTSOGNING.xlsx')
wos = pd.read_excel('WOS_SLUTSOGNING.xlsx')
curis = pd.read_excel('CURIS_VAL-UDG.xlsx')

comb_wopus = pd.concat([scopus, wos]).drop_duplicates(subset=['DOI'])

truant_curis = comb_wopus.loc[~comb_wopus['DOI'].isin(curis['DOI'])]
truant_curis.to_excel('slutsogning_22_mangler-i-curis.xlsx', index=False)
