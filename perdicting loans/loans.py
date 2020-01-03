import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 5000)

'''
# skip row 1 so pandas can parse the data properly.
loans_2007 = pd.read_csv('jaypeedevlin-lending-club-loan-data-2007-11/data/lending_club_loans.csv', low_memory=False)
half_count = len(loans_2007) / 2
na = (loans_2007.isnull().sum() / len(loans_2007)) * 100

loans_2007.dropna(thresh=half_count, axis=1, inplace=True)  # Drop any column with more than 50% missing values
loans_2007.drop(['url', 'desc'], axis=1, inplace=True)

# load the dictionary to understand the data
data_dictionary = pd.read_csv('jaypeedevlin-lending-club-loan-data-2007-11/data/lcdatadictionary.csv')

data_dictionary = data_dictionary.rename(columns={'LoanStatNew': 'name', 'Description': 'description'})

drop_list = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv',
             'int_rate', 'sub_grade', 'emp_title', 'issue_d', 'zip_code', 'out_prncp', 'out_prncp_inv',
             'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
             'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'zip_code', 'out_prncp', 'out_prncp_inv',
             'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
             'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
             'fico_range_high', 'fico_range_low']

loans_2007['fico_average'] = (loans_2007['fico_range_high'] + loans_2007['fico_range_low']) / 2

loans_2007 = loans_2007.drop(drop_list, axis=1)
loans_2007['loan_status'] = loans_2007['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

loans_2007 = loans_2007.loc[:, loans_2007.apply(pd.Series.nunique) != 1]
print(loans_2007)




loans_2007.drop(['tax_liens', 'pymnt_plan', 'acc_now_delinq'], axis=1, inplace=True)


for col in loans_2007:
	if (len(loans_2007[col].unique()) < 4):
		print(loans_2007[col].value_counts())
		print()

loans_2007.to_csv("filtered_loans_2007.csv",index=False)
'''
'''
filtered_loans = pd.read_csv('jaypeedevlin-lending-club-loan-data-2007-11/data/filtered_loans_2007.csv')

# PreProcessing the data #

# dealing with missing data
filtered_loans.dropna(axis=0, inplace=True)
# print((filtered_loans.isnull().sum() / len(filtered_loans)) * 100)

# dealing with categorical data
filtered_loans.drop(['last_credit_pull_d', 'addr_state', 'title', 'earliest_cr_line'], axis=1, inplace=True)

filtered_loans['emp_length'] = filtered_loans['emp_length'].map({"10+ years": 10, "9 years": 9, "8 years": 8,
                                                                 "7 years": 7,
                                                                 "6 years": 6,
                                                                 "5 years": 5,
                                                                 "4 years": 4,
                                                                 "3 years": 3,
                                                                 "2 years": 2,
                                                                 "1 year": 1,
                                                                 "< 1 year": 0,
                                                                 "n/a": 0
                                                                 })

filtered_loans['grade'] = filtered_loans['grade'].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})

mapping_purpose = {'car': 'Basics', 'medical': 'Basics', 'educational': 'Basics', 'house': 'Basics',
                   'debt_consolidation': 'Basics', 'credit_card': 'Basics',
                   'small_business': 'Luxuries', 'wedding': 'Luxuries', 'major_purchase': 'Luxuries',
                   'renewable_energy': 'Luxuries', 'vacation': 'Luxuries', 'moving': 'Luxuries',
                   'home_improvement': 'Luxuries', 'other': 'Others'}
filtered_loans.purpose = filtered_loans.purpose.map(mapping_purpose)

nominal_columns = ["home_ownership", "verification_status", "purpose", "term"]


obj = filtered_loans.select_dtypes(include='object')
print(filtered_loans.head())
for f in filtered_loans.columns:
    if filtered_loans[f].dtype=='object':
        lbl = LabelEncoder()
        lbl.fit(list(filtered_loans[f].values))
        filtered_loans[f] = lbl.transform(list(filtered_loans[f].values))

filtered_loans.to_csv("cleaned_loans_2007.csv",index=False)

'''
# loading the data

df = pd.read_csv('cleaned_loans_2007.csv')

y = df.loan_status
X = df.drop('loan_status', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model2 = SVC(C=20, gamma=3)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print(f1_score(y_test, y_pred))
