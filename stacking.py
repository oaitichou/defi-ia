# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:48:12 2018

"""

import pandas as pd

temp1 = pd.read_csv('C:/Users/Aitichou/Desktop/defibigdata/soumission/template_answer.csv', sep = ';')
temp2 = pd.read_csv('C:/Users\Aitichou/Desktop/defibigdata/template_answer_20Dec_2.csv', sep = ';')
temp3 = pd.read_csv('C:/Users/Aitichou/Desktop/defibigdata/soumission/template_answer_16Nov.csv', sep = ';')
temp4 = pd.read_csv('C:/Users/Aitichou/Desktop/defibigdata/soumission/template_answer_16Nov_2.csv', sep = ';')
temp5 = pd.read_csv('C:/Users/Aitichou/Desktop/defibigdata/template_answer_18Dec.csv', sep = ';')
temp6 = pd.read_csv('C:/Users\Aitichou/Desktop/defibigdata/prediction_stacking_07janvier_maj.csv', sep = ';')

dt = pd.DataFrame({'index' : temp1.name, 'pred1' : temp1.prediction, 'pred2' : temp2.prediction,'pred3' : temp3.prediction,
                   'pred4' : temp4.prediction, 'pred5' : temp5.prediction, 'pred6' : temp6.prediction})
dt

somme = dt[['pred1', 'pred2','pred3', 'pred4', 'pred5','pred6']].sum(axis=1)
dt['som'] = somme
dt['maj'] = [1 if x >= 4 else 0 for x in somme]
dtt = dt[dt.pred1 != dt.maj]
dtt.shape[0]

#dtt = dt[(dt.pred1 != dt.maj) & (dt['som'] == 3)]
#dtt = dt[dt.pred1 != dt.maj]
#dtt = dt[(dt.pred1 == dt.pred3) & (dt.pred1 != dt.pred2)]
#indx = list(dtt.index)
#temp1.prediction = [0 if x in indx else list(temp1.iloc[[x]].prediction)[0] for x in temp1.index]

temp1.prediction = dt.maj
temp1.to_csv('prediction_stacking_13janvier_essai.csv', sep = ';')

