"""
Bank data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sklearn.metrics as metrics


clean_accepted = 'data/clean_accepted.csv'
df_accepted = pd.read_csv(clean_accepted, low_memory=False)

#%%
"""
Label encoding

"""
print("Data types and their frequency\n{}".format(df_accepted.dtypes.value_counts()))

object_cols = df_accepted.select_dtypes(include=['object'])
print(object_cols.iloc[0])

for c in object_cols.columns:
    print(f'{c}:')
    print(object_cols[c].value_counts(), '\n')
    
drop_date_cols = ['earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'issue_d']
df_accepted = df_accepted.drop(drop_date_cols, axis=1)
# Drop addr_state becuase of too many values
df_accepted = df_accepted.drop('addr_state', axis=1)


grade_mappings = {'grade':{'A': 1,'B': 2,'C': 3,'D': 4,'E': 5,'F': 6,'G': 7}}
df_accepted = df_accepted.replace(grade_mappings)

nominal_cols = ['term', 'home_ownership', 'verification_status', 'purpose',
                'initial_list_status', 'debt_settlement_flag']
dummy_df = pd.get_dummies(df_accepted[nominal_cols])
df_accepted = pd.concat([df_accepted, dummy_df], axis=1)
df_accepted = df_accepted.drop(nominal_cols, axis=1)


df_accepted.info()

#%%

df1 = df_accepted.loc[:, 'loan_amnt':'revol_bal']
# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = df1.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Loan Attributes Correlation Heatmap', fontsize=12)

#%%
 #df_accepted = df_accepted.reset_index()
 #df_accepted.dropna(how='any', inplace=True)

X = df_accepted.loc[:, df_accepted.columns != 'loan_status']
y = df_accepted['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
scaler = MinMaxScaler()

# Apply scaling for algorithmns that used Eucledian distance
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
"""
Performing Logistic Regression

"""
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))

probs_log = logreg.predict_proba(X_test)
preds_log = probs_log[:,1]
fpr_log, tpr_log, threshold_log = metrics.roc_curve(y_test, preds_log)
roc_auc_log = metrics.auc(fpr_log, tpr_log)

plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.plot(fpr_log, tpr_log, 'b', label = 'AUC = %0.2f' % roc_auc_log)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%
"""
Performing Linear Discriminant Analysis

"""
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))

probs_log = logreg.predict_proba(X_test)
preds_log = probs_log[:,1]
fpr_log, tpr_log, threshold_log = metrics.roc_curve(y_test, preds_log)
roc_auc_log = metrics.auc(fpr_log, tpr_log)

plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.plot(fpr_log, tpr_log, 'b', label = 'AUC = %0.2f' % roc_auc_log)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
