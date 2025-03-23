import os
import logging
import csv
import datetime
from flask import Flask, render_template, redirect, url_for, flash, request, session, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
import json
import temp_auth
import models
from forms import LoginForm, SignupForm, PredictionForm

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Check if MongoDB is configured
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
db_initialized = False

# Initialize MongoDB if connection string exists
try:
    db = models.initialize_db(MONGO_URI)
    db_initialized = True
    logging.info("MongoDB initialized successfully")
    
    # Set authentication module based on database connection
    auth = models
    logging.info("Using MongoDB for authentication and data storage")
except Exception as e:
    logging.error(f"Error initializing MongoDB: {str(e)}")
    logging.info("Using temporary file-based authentication as fallback")
    db_initialized = False
    
    # Use temp_auth for authentication as fallback
    auth = temp_auth
    logging.info("Using file-based authentication as fallback")

# Configure file upload
ALLOWED_EXTENSIONS = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        # Check if username already exists
        existing_user = auth.get_user_by_username(form.username.data)
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('signup.html', form=form)
        
        # Create the user
        hashed_password = generate_password_hash(form.password.data)
        result = auth.create_user(form.username.data, hashed_password)
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = auth.get_user_by_username(form.username.data)
        
        if user and check_password_hash(user['password'], form.password.data):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('model', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    # Get user models
    user_models = auth.get_models_by_user(session['user_id'])
    
    # Get user predictions
    user_predictions = auth.get_predictions_by_user(session['user_id'])
    
    return render_template('dashboard.html', models=user_models, predictions=user_predictions)

@app.route('/model_analysis', methods=['GET', 'POST'])
def model_analysis():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if file is in the request
        if 'file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user doesn't select a file
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            try:
                # Read and process the CSV
                df = pd.read_csv(file)
                
                # Basic validation
                required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    flash(f'Missing required columns: {", ".join(missing_columns)}', 'danger')
                    return redirect(request.url)
                
                # Train the model
                X = df.drop('Outcome', axis=1)
                y = df['Outcome']
                
                # Get selected algorithm
                algorithm = request.form.get('algorithm', 'random_forest')
                
                # Data preprocessing
                # Handle missing values
                X = X.fillna(X.mean())
                
                # Feature scaling
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                
                X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
                
                # Select algorithm based on user choice
                if algorithm == 'random_forest':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif algorithm == 'logistic_regression':
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(random_state=42, max_iter=1000)
                elif algorithm == 'svm':
                    from sklearn.svm import SVC
                    model = SVC(probability=True, random_state=42)
                elif algorithm == 'gradient_boosting':
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(random_state=42)
                elif algorithm == 'knn':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier(n_neighbors=5)
                else:
                    # Default to Random Forest
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                
                # Make predictions and calculate metrics
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                cm = confusion_matrix(y_test, predictions)
                
                # Generate confusion matrix plot
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['No Diabetes', 'Diabetes'],
                            yticklabels=['No Diabetes', 'Diabetes'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                
                # Save plot to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                plt.close()
                
                # Feature importance plot - only for models that support it
                fi_image = None
                feature_importance_df = None
                
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': feature_importance
                    }).sort_values(by='Importance', ascending=False)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
                    plt.title('Feature Importance')
                    
                    # Save plot to a bytes buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    fi_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()
                    plt.close()
                elif algorithm == 'logistic_regression':
                    # For logistic regression, use coefficients as feature importance
                    coef = model.coef_[0]
                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': np.abs(coef)
                    }).sort_values(by='Importance', ascending=False)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
                    plt.title('Feature Importance (Coefficient Magnitude)')
                    
                    # Save plot to a bytes buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    fi_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()
                    plt.close()
                
                # Save model to file
                model_filename = f"model_{session['user_id']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
                joblib.dump(model, model_filename)
                
                # Store model info in database
                result = auth.save_model_data(
                    session['user_id'],
                    filename,
                    model_filename,
                    float(accuracy),
                    cm.tolist(),
                    list(X.columns),
                    algorithm
                )
                model_id = result['inserted_id']
                
                # Store model in session for prediction
                session['model'] = {
                    'id': str(model_id),
                    'filename': model_filename,
                    'feature_names': list(X.columns)
                }
                
                flash(f'Model trained successfully with accuracy: {accuracy:.2f}', 'success')
                
                # Prepare template context
                context = {
                    'accuracy': accuracy,
                    'cm_image': cm_image,
                    'fi_image': fi_image,
                    'algorithm': algorithm,
                    'feature_importance': feature_importance_df.to_dict('records') if feature_importance_df is not None else None
                }
                
                return render_template('model_analysis.html', **context)
            
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                logging.error(f"Error processing file: {str(e)}")
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload a CSV file.', 'danger')
            return redirect(request.url)
    
    return render_template('model_analysis.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    form = PredictionForm()
    
    # Get the latest model if none is in session
    if 'model' not in session:
        # Get the user's models and take the first one (most recent)
        user_models = auth.get_models_by_user(session['user_id'])
        latest_model = user_models[0] if user_models else None
        
        if latest_model:
            session['model'] = {
                'id': str(latest_model['_id']),
                'filename': latest_model['model_filename'],
                'feature_names': latest_model['feature_names']
            }
        else:
            flash('Please train a model first.', 'warning')
            return redirect(url_for('model_analysis'))
    
    if form.validate_on_submit():
        try:
            # Load the model
            model = joblib.load(session['model']['filename'])
            
            # Prepare input data
            input_data = {
                'Pregnancies': form.pregnancies.data,
                'Glucose': form.glucose.data,
                'BloodPressure': form.blood_pressure.data,
                'SkinThickness': form.skin_thickness.data,
                'Insulin': form.insulin.data,
                'BMI': form.bmi.data,
                'DiabetesPedigreeFunction': form.diabetes_pedigree_function.data,
                'Age': form.age.data
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of being diabetic
            
            # Save prediction to database
            auth.save_prediction(
                session['user_id'],
                session['model']['id'],
                input_data,
                int(prediction),
                float(prediction_proba)
            )
            
            result = {
                'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
                'probability': f"{prediction_proba * 100:.2f}%"
            }
            
            return render_template('prediction.html', form=form, result=result)
        
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'danger')
            logging.error(f"Error making prediction: {str(e)}")
    
    return render_template('prediction.html', form=form)

@app.route('/api/models')
def api_models():
    if 'user_id' not in session:
        return json.dumps({'error': 'Unauthorized'}), 401
    
    user_models = auth.get_models_by_user(session['user_id'])
    for model in user_models:
        model['_id'] = str(model['_id'])
        if isinstance(model['created_at'], str):
            # If already a string, keep as is
            pass
        else:
            model['created_at'] = model['created_at'].isoformat()
    
    return json.dumps(user_models)

@app.route('/api/predictions')
def api_predictions():
    if 'user_id' not in session:
        return json.dumps({'error': 'Unauthorized'}), 401
    
    user_predictions = auth.get_predictions_by_user(session['user_id'])
    for pred in user_predictions:
        pred['_id'] = str(pred['_id'])
        pred['model_id'] = str(pred['model_id'])
        if isinstance(pred['created_at'], str):
            # If already a string, keep as is
            pass
        else:
            pred['created_at'] = pred['created_at'].isoformat()
    
    return json.dumps(user_predictions)

@app.route('/export_predictions/<format>')
def export_predictions(format):
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    user_predictions = auth.get_predictions_by_user(session['user_id'])
    
    if not user_predictions:
        flash('No predictions to export.', 'warning')
        return redirect(url_for('dashboard'))
    
    # Create simplified predictions for export
    export_data = []
    for pred in user_predictions:
        # Convert ObjectId to string
        pred_id = str(pred['_id'])
        model_id = str(pred['model_id'])
        
        # Format the prediction data
        export_item = {
            'id': pred_id,
            'model_id': model_id,
            'pregnancies': pred['input_data']['Pregnancies'],
            'glucose': pred['input_data']['Glucose'],
            'blood_pressure': pred['input_data']['BloodPressure'],
            'skin_thickness': pred['input_data']['SkinThickness'],
            'insulin': pred['input_data']['Insulin'],
            'bmi': pred['input_data']['BMI'],
            'diabetes_pedigree_function': pred['input_data']['DiabetesPedigreeFunction'],
            'age': pred['input_data']['Age'],
            'prediction': 'Diabetic' if pred['prediction'] == 1 else 'Non-Diabetic',
            'probability': f"{pred['prediction_probability'] * 100:.2f}%",
            'date': pred['created_at'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(pred['created_at'], 'strftime') else pred['created_at']
        }
        export_data.append(export_item)
    
    if format == 'csv':
        # Convert to CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(export_data[0].keys())
        
        # Write data
        for item in export_data:
            writer.writerow(item.values())
        
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=predictions_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"}
        )
    
    elif format == 'json':
        # Return as JSON file
        return Response(
            json.dumps(export_data, indent=4),
            mimetype="application/json",
            headers={"Content-disposition": f"attachment; filename=predictions_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"}
        )
    
    else:
        flash('Invalid export format.', 'danger')
        return redirect(url_for('dashboard'))

@app.context_processor
def utility_processor():
    def is_user_logged_in():
        return 'user_id' in session
    
    def get_username():
        return session.get('username', '')
    
    return {
        'is_user_logged_in': is_user_logged_in,
        'get_username': get_username
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
