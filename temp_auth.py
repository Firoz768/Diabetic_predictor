import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# File to store user data temporarily
USERS_FILE = "temp_users.json"
MODELS_FILE = "temp_models.json"
PREDICTIONS_FILE = "temp_predictions.json"

def _load_users():
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def _save_users(users):
    """Save users to file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def _load_models():
    """Load models from file"""
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, 'r') as f:
            return json.load(f)
    return []

def _save_models(models):
    """Save models to file"""
    with open(MODELS_FILE, 'w') as f:
        json.dump(models, f, indent=4)

def _load_predictions():
    """Load predictions from file"""
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def _save_predictions(predictions):
    """Save predictions to file"""
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=4)

def _generate_id():
    """Generate a simple ID"""
    import uuid
    return str(uuid.uuid4())

# User functions
def create_user(username, password_hash):
    """Create a new user"""
    users = _load_users()
    if username in users:
        return None
    
    user_id = _generate_id()
    users[username] = {
        '_id': user_id,
        'username': username,
        'password': password_hash,
        'created_at': datetime.now().isoformat()
    }
    _save_users(users)
    return {'inserted_id': user_id}

def get_user_by_username(username):
    """Get a user by username"""
    users = _load_users()
    if username in users:
        return users[username]
    return None

def get_user_by_id(user_id):
    """Get a user by ID"""
    users = _load_users()
    for user in users.values():
        if user['_id'] == user_id:
            return user
    return None

# Model functions
def save_model_data(user_id, filename, model_filename, accuracy, confusion_matrix, feature_names, algorithm=None):
    """Save model data"""
    models = _load_models()
    model_id = _generate_id()
    
    model = {
        '_id': model_id,
        'user_id': user_id,
        'filename': filename,
        'model_filename': model_filename,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'feature_names': feature_names,
        'algorithm': algorithm,
        'created_at': datetime.now().isoformat()
    }
    
    models.append(model)
    _save_models(models)
    return {'inserted_id': model_id}

def get_models_by_user(user_id):
    """Get all models for a user"""
    models = _load_models()
    return [model for model in models if model['user_id'] == user_id]

def get_model_by_id(model_id):
    """Get a model by ID"""
    models = _load_models()
    for model in models:
        if model['_id'] == model_id:
            return model
    return None

# Prediction functions
def save_prediction(user_id, model_id, input_data, prediction, prediction_probability):
    """Save a prediction"""
    predictions = _load_predictions()
    prediction_id = _generate_id()
    
    prediction_data = {
        '_id': prediction_id,
        'user_id': user_id,
        'model_id': model_id,
        'input_data': input_data,
        'prediction': prediction,
        'prediction_probability': prediction_probability,
        'created_at': datetime.now().isoformat()
    }
    
    predictions.append(prediction_data)
    _save_predictions(predictions)
    return {'inserted_id': prediction_id}

def get_predictions_by_user(user_id):
    """Get all predictions for a user"""
    predictions = _load_predictions()
    return [pred for pred in predictions if pred['user_id'] == user_id]