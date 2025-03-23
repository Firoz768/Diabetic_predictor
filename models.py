import os
from datetime import datetime
import logging

# Initialize with default empty collections
users_collection = None
models_collection = None
predictions_collection = None
db = None

# We'll initialize these properly when the MongoDB URI is provided
try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    # Only attempt MongoDB connection if URI is set
    mongo_uri = os.environ.get("MONGO_URI")
    if mongo_uri:
        client = MongoClient(mongo_uri)
        db = client.get_database()
        
        # Set up collections
        users_collection = db.users
        models_collection = db.models
        predictions_collection = db.predictions
        logging.info("MongoDB connected successfully")
except Exception as e:
    logging.error(f"MongoDB connection error: {str(e)}")
    # Continue with empty collections - we'll initialize them when the connection string is provided

def initialize_db(mongo_uri):
    """Initialize the MongoDB connection with the provided URI"""
    global client, db, users_collection, models_collection, predictions_collection
    
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    
    try:
        client = MongoClient(mongo_uri)
        db = client.diabetes_predictor  # Explicitly set database name
        
        # Set up collections
        users_collection = db.users
        models_collection = db.models
        predictions_collection = db.predictions
        
        logging.info("MongoDB connected successfully")
        return db
    except Exception as e:
        logging.error(f"MongoDB connection error: {str(e)}")
        return None

# User model functions
def create_user(username, password_hash):
    """Create a new user in the database"""
    if users_collection is None:
        logging.error("Cannot create user: MongoDB not connected")
        return None
        
    user_data = {
        'username': username,
        'password': password_hash,
        'created_at': datetime.now()
    }
    return users_collection.insert_one(user_data)

def get_user_by_username(username):
    """Get a user by username"""
    if users_collection is None:
        logging.error("Cannot get user: MongoDB not connected")
        return None
        
    return users_collection.find_one({'username': username})

def get_user_by_id(user_id):
    """Get a user by ID"""
    if users_collection is None:
        logging.error("Cannot get user: MongoDB not connected")
        return None
    
    try:
        from bson.objectid import ObjectId
        return users_collection.find_one({'_id': ObjectId(user_id)})
    except Exception as e:
        logging.error(f"Error getting user by ID: {str(e)}")
        return None

# Model functions
def save_model_data(user_id, filename, model_filename, accuracy, confusion_matrix, feature_names, algorithm=None):
    """Save model data to the database"""
    if models_collection is None:
        logging.error("Cannot save model data: MongoDB not connected")
        return None
        
    model_data = {
        'user_id': user_id,
        'filename': filename,
        'model_filename': model_filename,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'feature_names': feature_names,
        'algorithm': algorithm,
        'created_at': datetime.now()
    }
    return models_collection.insert_one(model_data)

def get_models_by_user(user_id):
    """Get all models for a user"""
    if models_collection is None:
        logging.error("Cannot get models: MongoDB not connected")
        return []
        
    return list(models_collection.find({'user_id': user_id}).sort('created_at', -1))

def get_model_by_id(model_id):
    """Get a model by ID"""
    if models_collection is None:
        logging.error("Cannot get model: MongoDB not connected")
        return None
    
    try:
        from bson.objectid import ObjectId
        return models_collection.find_one({'_id': ObjectId(model_id)})
    except Exception as e:
        logging.error(f"Error getting model by ID: {str(e)}")
        return None

# Prediction functions
def save_prediction(user_id, model_id, input_data, prediction, prediction_probability):
    """Save a prediction to the database"""
    if predictions_collection is None:
        logging.error("Cannot save prediction: MongoDB not connected")
        return None
        
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
    if predictions_collection is None:
        logging.error("Cannot get predictions: MongoDB not connected")
        return []
        
    return list(predictions_collection.find({'user_id': user_id}).sort('created_at', -1))
