import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('data.csv')

co_threshold = df['CO'].quantile(0.90)
nox_threshold = df['NOX'].quantile(0.90)
tit_threshold = df['TIT'].quantile(0.95)

df['high_CO'] = (df['CO'] > co_threshold).astype(int)
df['high_NOX'] = (df['NOX'] > nox_threshold).astype(int)
df['high_TIT'] = (df['TIT'] > tit_threshold).astype(int)
df['failure'] = (df['high_CO'] | df['high_NOX'] | df['high_TIT']).astype(int)


failure_count = df['failure'].sum()
failure_rate = failure_count / len(df) * 100


df['failure_in_24h'] = df['failure'].shift(-24).fillna(0)


# clean the data

df = df.dropna()



# prepare for predictions


features = [col for col in df.columns if col not in ['failure_in_24h', 'failure', 'high_CO', 'high_NOX', 'high_TIT']]
X = df[features]
y = df['failure_in_24h']


# train the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# train the model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predict the model

y_pred = model.predict(X_test)

# evaluate the model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



latest = X_test.iloc[-1:].copy()
prob = model.predict_proba(latest)[0][1]

if prob > 0.7:
    print(f"\n⚠️  High risk of failure in the next 24 hours! (Probability: {prob:.2f})")
elif prob > 0.3:
    print(f"\n⚠️  Moderate risk of failure in the next 24 hours. (Probability: {prob:.2f})")
else:
    print(f"\n✅ Low risk of failure in the next 24 hours. (Probability: {prob:.2f})")

# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Failure'],
            yticklabels=['Normal', 'Failure'])
plt.title('Confusion Matrix - Model Performance')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Feature Importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.barh(range(len(feature_importance.head(10))), feature_importance.head(10)['importance'])
plt.yticks(range(len(feature_importance.head(10))), feature_importance.head(10)['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
