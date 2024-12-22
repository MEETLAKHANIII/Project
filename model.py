import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelPipeline:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def preprocess_data(self):
        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        # Encode categorical variables
        label_encoders = {}
        for col in self.X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
            label_encoders[col] = le

        # Encode target variable if necessary
        if self.y.dtype == 'object':
            target_encoder = LabelEncoder()
            self.y = target_encoder.fit_transform(self.y)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self, model):
        self.model = model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Calculate evaluation metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        return metrics

# Example Usage
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("/mnt/data/project data.csv")

    # Create pipeline object
    pipeline = ModelPipeline(data=data, target_column='Response')

    # Preprocess data
    pipeline.preprocess_data()

    # Train Logistic Regression model
    lr_model = LogisticRegression(random_state=42)
    pipeline.train_model(lr_model)

    # Evaluate model
    metrics = pipeline.evaluate_model()
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
