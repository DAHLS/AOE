# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:17:56 2024

@author: asda
"""

import pandas as pd
import glob

wos_files = glob.glob('wosparts/*.xls')
excl_parts = list()
excl_merg = pd.DataFrame()

for f in wos_files:
    excl_parts.append(pd.read_excel(f))
    
for excl_file in excl_parts:
    excl_merg = excl_merg.append(excl_file, ignore_index=True)

    
with pd.ExcelWriter("wos.xlsx") as writer:
    excl_merg.to_excel(writer)