# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:38:52 2020

@author: admin
"""
import os
path=os.getcwd()+'\\'
def Auto_Name(name_of_py):
    return 'tf_upgrade_v2 --infile '+str(path)+str(name_of_py)+'-TF1.py --outfile '+str(path)+str(name_of_py)+'-TF2.py --reportfile '+str(path)+'reports'+str(name_of_py)+'.txt' 
name_of_py='code6-18'
print(Auto_Name(name_of_py))