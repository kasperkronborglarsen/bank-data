"""
Bank data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


accepted_loans_raw = 'data/accepted_2007_to_2018Q4.csv'
df_accepted = pd.read_csv(accepted_loans_raw, low_memory=False)
print(f'Raw shape: {df_accepted.shape}') #(2260701, 151)

# Drop columns with more than 50% missing values
df_accepted = df_accepted.dropna(thresh=len(df_accepted)/2, axis=1)
print(f'Processed shape: {df_accepted.shape}') #(2260701, 107)


"""
Narrowing down the columns

"""

drop_list = ['id', 'title', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'emp_title', 'emp_length', 'pymnt_plan', 'url', 'zip_code', 'application_type', 'disbursement_method']

df_accepted = df_accepted.drop(drop_list, axis=1)

# Drop columns with containing only one unique value
for col in df_accepted.columns:
    if (len(df_accepted[col].unique()) < 2):
        print(df_accepted[col].value_counts())
        print() #out_prncp, out_prncp_inv, policy_code, hardship_flag
        
second_drop_list = ['out_prncp', 'out_prncp_inv', 'policy_code', 'hardship_flag']
df_accepted = df_accepted.drop(second_drop_list, axis=1)

# Remove columns with more than 1% missing values
null_count = df_accepted.isnull().sum()
null_cols = []
for i in range(len(null_count)):
    if null_count[i] > (df_accepted.shape[0]/100):
        null_cols.append(null_count.index[i])
    
    
print(null_cols)

df_accepted = df_accepted.drop(null_cols, axis=1)

#df_accepted = df_accepted.reset_index()
df_accepted.dropna(how='any', inplace=True)

#%%
"""
Choosing target value

"""

df_accepted = df_accepted.loc[(df_accepted['loan_status'] == 'Fully Paid') | (df_accepted['loan_status'] == 'Charged Off')]

status_mappings = {"loan_status":{ "Fully Paid": 1, "Charged Off": 0}}
df_accepted = df_accepted.replace(status_mappings)

#paid_loans = df_accepted.loc[(df_accepted['loan_status'] == 1)]
#chargedOff_loans = df_accepted.loc[(df_accepted['loan_status'] == 0)]

dims = (8, 6)
fig, ax = plt.subplots(figsize=dims)
sns.countplot(ax=ax, x='loan_status',data=df_accepted).set_title('Frequency of the Loan Status')

df_accepted.to_csv('data/clean_accepted.csv', index=False)