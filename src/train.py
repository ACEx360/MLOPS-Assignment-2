import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():

    # Set MLflow experiment
    mlflow.set_experiment("Titanic_Classification")

    # Start MLflow run
    with mlflow.start_run(run_name="Logistic_Regression"):

        # Load dataset
        df = pd.read_csv("data/Titanic-Dataset.csv")

        print("Dataset Shape:", df.shape)

        # Drop unnecessary columns
        df.drop(
            columns=["PassengerId", "Name", "Ticket", "Cabin"],
            inplace=True,
        )

        # Fill missing values
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        # Encode categorical columns
        encoder = LabelEncoder()

        for column in ["Sex", "Embarked"]:
            df[column] = encoder.fit_transform(df[column])

        # Features and target
        X = df.drop("Survived", axis=1)
        y = df["Survived"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Model parameters
        max_iter = 200
        random_state = 42

        # Log parameters
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)

        # Train model
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )

        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Print metrics
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)

        fig, ax = plt.subplots()

        ConfusionMatrixDisplay(
            confusion_matrix=cm
        ).plot(ax=ax)

        plt.title("Logistic Regression - Confusion Matrix")
        plt.tight_layout()

        confusion_matrix_path = "logistic_regression_confusion_matrix.png"

        plt.savefig(confusion_matrix_path)
        plt.close()

        mlflow.log_artifact(confusion_matrix_path)

        # Classification Report
        report = classification_report(
            y_test,
            predictions
        )

        print("\nClassification Report:")
        print(report)

        report_path = "logistic_regression_classification_report.txt"

        with open(report_path, "w") as file:
            file.write(report)

        mlflow.log_artifact(report_path)

        # Save model
        model_path = "models/titanic_logistic_regression.pkl"

        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="Titanic_Logistic_Regression",
        )

        print("\nModel saved in models/titanic_logistic_regression.pkl")
        print("Experiment tracked in MLflow")


if __name__ == "__main__":
    main()