import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

file_path = 'StudentsPerformance.csv'
data = pd.read_csv(file_path)

encoded_data = data.copy()
label_encoders = {}
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for col in categorical_columns:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

score_columns = ['math score', 'reading score', 'writing score']
X = encoded_data.drop(score_columns, axis=1)
y = encoded_data['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=X.columns, filled=True, fontsize=10)
plt.title("Arbre de décision pour prédire les scores en mathématiques")
plt.show()
