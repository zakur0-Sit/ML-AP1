import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.naive_bayes import GaussianNB

file_path = 'grafic/Grafic_SEN.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse("Grafic SEN")
df['Data'] = pd.to_datetime(df['Data'], errors='coerce', dayfirst=True)

cleaned_df = df.dropna()
cleaned_df = cleaned_df[cleaned_df[['Foto[MW]', 'Consum[MW]', 'Productie[MW]']].min(axis=1) >= 0]

features = ['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
            'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]',
            'Foto[MW]', 'Biomasa[MW]']
target = 'Sold[MW]'

X = cleaned_df[features]
y = cleaned_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

id3_model = DecisionTreeRegressor(random_state=25)
id3_model.fit(X_train, y_train)

y_pred = id3_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE for Decision Tree: {rmse}")

num_bins = 10
bins = np.linspace(y_train.min(), y_train.max(), num_bins)
y_train_binned = np.digitize(y_train, bins)
y_test_binned = np.digitize(y_test, bins)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train_binned)

y_test_pred_binned = nb_model.predict(X_test)

bin_centers = (bins[:-1] + bins[1:]) / 2
y_test_pred_continuous = bin_centers[y_test_pred_binned - 1]

rmse_nb = np.sqrt(mean_squared_error(y_test, y_test_pred_continuous))
print(f"RMSE for Naive Bayes: {rmse_nb}")

december_data = cleaned_df[cleaned_df['Data'].dt.month == 12]
december_data['Data'] = december_data['Data'].apply(lambda x: x.replace(year=2024))

daily_data = december_data.groupby(december_data['Data'].dt.date).mean()

X_daily = daily_data[features]

daily_predictions_id3 = id3_model.predict(X_daily)

daily_predictions_nb = nb_model.predict(X_daily)
daily_predictions_nb_continuous = bin_centers[daily_predictions_nb - 1]

daily_data['Predicted Sold[MW] - Decision Tree'] = daily_predictions_id3
daily_data['Predicted Sold[MW] - Naive Bayes'] = daily_predictions_nb_continuous

output_file = 'Predicted_December_Sold.xlsx'
daily_data[['Predicted Sold[MW] - Decision Tree', 'Predicted Sold[MW] - Naive Bayes']].to_excel(output_file, sheet_name='December Predictions', index=True)

print(f"Predictions for December 2024 saved to {output_file}")