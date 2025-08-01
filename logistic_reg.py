import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

df = pd.read_csv("/content/sample_data/CustomersDataSet")

df = df[df['Profession'].isin(['Engineer', 'Doctor'])].copy()

df['HighSpender'] = (df['Spending Score (1-100)'] > 50).astype(int)
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
df['Profession_encoded'] = le_profession.fit_transform(df['Profession'])

features = ['Profession_encoded', 'Gender_encoded', 'Age', 'Annual Income ($)', 'Work Experience', 'Family Size']
X = df[features]
y = df['HighSpender']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()