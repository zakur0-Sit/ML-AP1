import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

file_path = 'grafic/Grafic_SEN.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse("Grafic SEN")
df['Data'] = pd.to_datetime(df['Data'], errors='coerce', dayfirst=True)

cleaned_df = df.dropna()
cleaned_df = cleaned_df[cleaned_df[['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']].min(axis=1) >= 0]

features = ['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']
target = 'Sold[MW]'

X = cleaned_df[features]
y = cleaned_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

id3_model = DecisionTreeRegressor(random_state=25)
id3_model.fit(X_train, y_train)

december_data = cleaned_df[cleaned_df['Data'].dt.month == 12]
december_data['Data'] = december_data['Data'].apply(lambda x: x.replace(year=2024))
daily_data = december_data.groupby(december_data['Data'].dt.date).mean()
X_daily = daily_data[features]
daily_predictions = id3_model.predict(X_daily)

results = daily_data.copy()
results['Predicted Sold[MW]'] = daily_predictions
results['Data'] = results.index
results = results[['Data', 'Predicted Sold[MW]']]

output_file = 'Predicted_December_2024_Sold.xlsx'
results.to_excel(output_file, sheet_name='December Predictions', index=False)

print(f"Predictions for December 2024 saved to {output_file}")