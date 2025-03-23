import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

def validate_dataset(df):
    """
    Validate if the dataset has required columns
    
    Args:
        df (pandas.DataFrame): The dataset to validate
        
    Returns:
        tuple: (is_valid, message)
    """
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for nulls
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        return False, f"Dataset contains {nulls} null values."
    
    # Check for outcome values (0 or 1)
    if not all(df['Outcome'].isin([0, 1])):
        return False, "Outcome column should only contain 0 (non-diabetic) or 1 (diabetic)."
    
    return True, "Dataset is valid."

def train_model(df):
    """
    Train a RandomForestClassifier on the dataset
    
    Args:
        df (pandas.DataFrame): The dataset to train on
        
    Returns:
        dict: Training results
    """
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    # Feature importance plot
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    fi_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    roc_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    return {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'feature_names': list(X.columns),
        'cm_image': cm_image,
        'fi_image': fi_image,
        'roc_image': roc_image,
        'feature_importance': feature_importance_df.to_dict('records')
    }

def format_prediction_result(prediction, probability):
    """
    Format prediction results for display
    
    Args:
        prediction (int): 0 or 1 representing non-diabetic or diabetic
        probability (float): Probability of being diabetic
        
    Returns:
        dict: Formatted prediction result
    """
    return {
        'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
        'probability': f"{probability * 100:.2f}%",
        'risk_level': get_risk_level(probability)
    }

def get_risk_level(probability):
    """
    Determine risk level based on probability
    
    Args:
        probability (float): Probability of being diabetic
        
    Returns:
        str: Risk level (Low, Moderate, High)
    """
    if probability < 0.3:
        return {'level': 'Low', 'class': 'text-success'}
    elif probability < 0.7:
        return {'level': 'Moderate', 'class': 'text-warning'}
    else:
        return {'level': 'High', 'class': 'text-danger'}
