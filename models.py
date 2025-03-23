import os
from datetime import datetime

# We'll initialize MongoDB connection later when the URI is provided
# Placeholder collections
users_collection = None
models_collection = None
predictions_collection = None

def initialize_db(mongo_uri):
    """Initialize the MongoDB connection with the provided URI"""
    from pymongo import MongoClient
    global users_collection, models_collection, predictions_collection
    
    client = MongoClient(mongo_uri)
    db = client.get_database()
    
    # Set up collections
    users_collection = db.users
    models_collection = db.models
    predictions_collection = db.predictions
    
    return db

# User model functions
def create_user(username, password_hash):
    """Create a new user in the database"""
    user_data = {
        'username': username,
        'password': password_hash,
        'created_at': datetime.now()
    }
    return users_collection.insert_one(user_data)

def get_user_by_username(username):
    """Get a user by username"""
    return users_collection.find_one({'username': username})

def get_user_by_id(user_id):
    """Get a user by ID"""
    return users_collection.find_one({'_id': user_id})

# Model functions
def save_model_data(user_id, filename, model_filename, accuracy, confusion_matrix, feature_names):
    """Save model data to the database"""
    model_data = {
        'user_id': user_id,
        'filename': filename,
        'model_filename': model_filename,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'feature_names': feature_names,
        'created_at': datetime.now()
    }
    return models_collection.insert_one(model_data)

def get_models_by_user(user_id):
    """Get all models for a user"""
    return list(models_collection.find({'user_id': user_id}).sort('created_at', -1))

def get_model_by_id(model_id):
    """Get a model by ID"""
    return models_collection.find_one({'_id': model_id})

# Prediction functions
def save_prediction(user_id, model_id, input_data, prediction, prediction_probability):
    """Save a prediction to the database"""
    prediction_data = {
        'user_id': user_id,
        'model_id': model_id,
        'input_data': input_data,
        'prediction': prediction,
        'prediction_probability': prediction_probability,
        'created_at': datetime.now()
    }
    return predictions_collection.insert_one(prediction_data)

def get_predictions_by_user(user_id):
    """Get all predictions for a user"""
    return list(predictions_collection.find({'user_id': user_id}).sort('created_at', -1))
