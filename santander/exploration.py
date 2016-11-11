import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

prod_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

def interpret_basket(string):
    results =  ''
    for a in string:
        if a == '1':
            if result == '':
                result = prod_cols[a]
            else:
                result = ' '.join(prod_cols[a])
    return result


data_types = {
                #'fecha_dato' : date,
                'ncodpers' : np.int8,
                'ind_empleado' : bool,
                'pais_residencia' : str,
                'sexo' : str,
                'age' : np.int8,
                #'fecha_alta' : date,
                'ind_nuevo' : bool,
                #'antiguedad' : ,
                #'indrel' : ,
                #'ult_fec_cli_1t' : ,
                # 'indrel_1mes' : ,
                # 'tiprel_1mes' : ,
                # 'indresi' : ,
                # 'indext' : ,
                # 'conyuemp' : ,
                # 'canal_entrada' : ,
                # 'indfall' : ,
                # 'tipodom' : ,
                'cod_prov' : np.int8,
                'nomprov' : str,
                #'ind_actividad_cliente' : ,
                'renta' : np.float64,
                'segmento' : str,
                'ind_ahor_fin_ult1' : np.int8,
                'ind_aval_fin_ult1' : np.int8,
                'ind_cco_fin_ult1' : np.int8,
                'ind_cder_fin_ult1': np.int8,
                'ind_cno_fin_ult1': np.int8,
                'ind_ctju_fin_ult1': np.int8,
                'ind_ctma_fin_ult1': np.int8,
                'ind_ctop_fin_ult1': np.int8,
                'ind_ctpp_fin_ult1': np.int8,
                'ind_deco_fin_ult1': np.int8,
                'ind_deme_fin_ult1': np.int8,
                'ind_dela_fin_ult1': np.int8,
                'ind_ecue_fin_ult1': np.int8,
                'ind_fond_fin_ult1': np.int8,
                'ind_hip_fin_ult1': np.int8,
                'ind_plan_fin_ult1': np.int8,
                'ind_pres_fin_ult1': np.int8,
                'ind_reca_fin_ult1': np.int8,
                'ind_tjcr_fin_ult1': np.int8,
                'ind_valo_fin_ult1': np.int8,
                'ind_viv_fin_ult1': np.int8,
                'ind_nomina_ult1': np.int8,
                'ind_nom_pens_ult1': np.int8,
                'ind_recibo_ult1' : np.int8}

#training cleanup
df_train = pd.read_csv('train_ver2.csv')
df_train[prod_cols] = df_train[prod_cols].fillna(0)
df_train[prod_cols] = df_train[prod_cols].astype(np.int8)
df_train['nomprov'].fillna('None Given',inplace=True)
df_train['nomprov'] = df_train['nomprov'].astype('category')

# clean up segments (train)
df_train.loc[df_train['segmento'] == '01 - TOP', 'segmento'] = 1
df_train.loc[df_train['segmento'] == '02 - PARTICULARES', 'segmento'] = 2
df_train.loc[df_train['segmento'] == '03 - UNIVERSITARIO', 'segmento'] = 3
df_train['segmento'].fillna(4,inplace=True)
df_train['segmento'] = df_train['segmento'].astype(np.int8)

# test cleanup
df_test = pd.read_csv('test_ver2.csv')
df_test[prod_cols] = df_train[prod_cols].fillna(0)
df_test[prod_cols] = df_train[prod_cols].astype(np.int8)
df_test['nomprov'].fillna('None Given',inplace=True)

# clean up segments (test)
df_test.loc[df_test['segmento'] == '01 - TOP', 'segmento'] = 1
df_test.loc[df_test['segmento'] == '02 - PARTICULARES', 'segmento'] = 2
df_test.loc[df_test['segmento'] == '03 - UNIVERSITARIO', 'segmento'] = 3
df_test['segmento'].fillna(4,inplace=True)
df_test['segmento'] = df_test['segmento'].astype(np.int8)

# fit random forest
rf_model = ensemble.RandomForestClassifier(verbose=1)

df_train[prod_cols] = df_train[prod_cols].astype(str)
df_train['customer_product_basket'] = df_train[prod_cols].apply(lambda x: ''.join(x), axis=1)
target = df_train['customer_product_basket'].values
df_train.drop('customer_product_basket',axis=1,inplace=True)
df_train.drop(prod_cols,axis=1,inplace=True)

data_array = df_train.values

rf_model_fitted = rf_model.fit(data_array,target)

test_array = df_test.values
prediction = lr_model_fitted.predict(test_array)

test_ids = df_test['ncodpers'].values

sample = df_test['ncodpers']

sample['pred'] = prediction
sample['added_products'] = df_train['prediction'].apply(interpret_basket)

submission = sample.set_index('ncodpers').to_csv("submission.csv")

del df_train
del test
del sample
del prediction
