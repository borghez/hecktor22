import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt

from sksurv.util            import Surv
from sksurv.metrics         import concordance_index_censored
from icare.survival         import BaggedIcareSurvival
from icare.visualisation    import plot_avg_sign
from icare.metrics          import harrell_cindex

path     = "C:\\Users\\andre\\Desktop\\OUTCOME PREDICTION\\"

df_train = pd.read_csv(path + "All_features\\all_features_hecktor_train.csv", index_col='PatientID')
#df_train = pd.read_csv('/home/andrea/Codici/NEW/WBall_features_train_new.csv', index_col='PatientID')

df_test = pd.read_csv(path + "All_features\\all_features_hecktor_test.csv", index_col='PatientID')
#df_test = pd.read_csv('/home/andrea/Codici/NEW/WBall_features_test_new.csv', index_col='PatientID')

y_test_true = Surv.from_arrays( event=df_test['Relapse'].values,
                                time=df_test['RFS'].values )

df_train.drop(['PatientID.1', 'X_size', 'Y_size', 'Z_size'], axis=1, inplace=True)
df_test.drop(['PatientID.1', 'X_size', 'Y_size', 'Z_size', 'Relapse', 'RFS'], axis=1, inplace=True)

df_train.Gender = df_train.Gender.map({'M': 1, 'F': 0})
df_test.Gender = df_test.Gender.map({'M': 1, 'F': 0})

for riga in range(df_test.shape[0]):
    for colonna in range(df_test.shape[1]):

        if isinstance(df_test.iloc[riga, colonna], str):
            temp = df_test.iloc[riga, colonna].replace('(', '')
            temp = temp.replace('+0j)', '')
            df_test.iloc[riga, colonna] = float(temp)

df_test.rename(columns={'Performance_status': 'Performance status'}, inplace=True)

print(df_train.shape), print(df_test.shape)


features = list(set(df_train.columns.tolist()) - {'Relapse', 'RFS', 'Task 1', 'Task 2', 'CenterID'})
features = [x for x in features if 'lesions_merged' not in x and 'lymphnodes_merged' not in x]
extra_features = ['Gender',
                  'Age',
                  'Weight',
                  'Tobacco',
                  'Alcohol',
                  'Performance status',
                  'HPV status (0=-, 1=+)',
                  'Surgery',
                  'Chemotherapy',
                  'nb_lesions',
                  'nb_lymphnodes',
                  'whole_body_scan'
                  ]

features_groups = np.unique([x.split('_shape_')[0].split('_PT_')[0].split('_CT_')[0] for x in features])
features_groups = list(set(features_groups) - set(extra_features))
features_groups = [x + '_' for x in features_groups]
features_groups.append('extra_features')
len(features_groups), features_groups

y_train = Surv.from_arrays(event=df_train['Relapse'].values,
                           time=df_train['RFS'].values)

X_train, X_test = df_train[features], df_test[features]


features_groups_id = []
for f in X_train.columns:
    if f in extra_features:
        features_groups_id.append(features_groups.index('extra_features'))
    else:
        group = f.split('_shape_')[0].split('_PT_')[0].split('_CT_')[0] + '_'
        features_groups_id.append(features_groups.index(group))

hyperparameters_sets = [
 {'rho': 0.66,
  'cmin': 0.53,
  'max_features': 0.00823045267489712,
  'mandatory_features': extra_features,
  'sign_method': 'harrell',
  'features_groups_to_use': [2, 4, 8, 10]},
 {'rho': 0.72,
  'cmin': 0.59,
  'max_features': 0.009465020576131687,
  'mandatory_features': extra_features,
  'sign_method': 'harrell',
  'features_groups_to_use': [3, 4, 10, 11, 12]},
 {'rho': 0.87,
  'cmin': 0.55,
  'max_features': 0.06131687242798354,
  'mandatory_features': extra_features,
  'sign_method': 'harrell',
  'features_groups_to_use': [1, 3, 4, 10, 11, 12]},
 {'rho': 0.57,
  'cmin': 0.51,
  'max_features': 0.005761316872427984,
  'mandatory_features': extra_features,
  'sign_method': 'harrell',
  'features_groups_to_use': [0, 2, 5, 6, 9, 11, 12]},
 {'rho': 0.71,
  'cmin': 0.57,
  'max_features': 0.16131687242798354,
  'mandatory_features': extra_features,
  'sign_method': 'harrell',
  'features_groups_to_use': [4, 8, 12]}
]


model = BaggedIcareSurvival(n_estimators=1000,
                            parameters_sets=hyperparameters_sets,
                            aggregation_method='median',
                            n_jobs=-1)
model.fit(X_train, y_train, feature_groups=features_groups_id)
test_pred = model.predict(X_test)

test_pred_dataframe = pd.DataFrame(test_pred)
test_pred_dataframe.to_csv('test_pred.csv')
test_pred_dataframe.to_csv(path + "Predictions\\hecktor_test_predictions.csv")

rad_extra = [
'everything_merged_shape_Maximum2DDiameterRow',
'everything_mergedBBox_CT_firstorder_90Percentile',
'everything_merged40%_PT_gldm_HighGrayLevelEmphasis',
'everything_mergedshell4mm_PT_gldm_LargeDependenceLowGrayLevelEmphasis',
'everything_mergeddilat16mm_PT_firstorder_Kurtosis',
'everything_mergeddilat4mm_shape_Flatness',
]

plot_avg_sign(model, features=extra_features + rad_extra)
plt.show()

def check_target(y):
    is_surv = False
    try:
        b = [x[1] for x in y]
        is_surv = True
    except:
        pass

    if not is_surv:
        return Surv.from_arrays(event=np.full(len(y), True),
                                time=y)
    return y


def harrell_cindex(y_true, y_pred):
    y_true = check_target(y_true)
    return concordance_index_censored(event_indicator=np.array([x[0] for x in y_true]).astype('bool'),
                                      event_time=np.array([x[1] for x in y_true]).astype('float32'),
                                      estimate=y_pred)

print(harrell_cindex(y_test_true, test_pred))

c_index, concordant_pairs, discordant_pairs, tied_risk, tied_time = harrell_cindex(y_test_true, test_pred)

results = { 'C-index':          [c_index],
            'Concordant pairs': [concordant_pairs],
            'Discordant pairs': [discordant_pairs],
            'Tied risk':        [tied_risk],
            'Tied time':        [tied_time]}

results = pd.DataFrame(results)
#results.index = ['Hecktor']

results.to_csv(path + "Results\\results_hecktor.csv")

