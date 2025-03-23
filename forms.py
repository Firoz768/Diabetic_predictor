from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FloatField, IntegerField
from wtforms.validators import DataRequired, Length, EqualTo, NumberRange

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=20, message="Username must be between 3 and 20 characters")
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6, message="Password must be at least 6 characters")
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message="Passwords must match")
    ])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=20)
    ])
    password = PasswordField('Password', validators=[
        DataRequired()
    ])
    submit = SubmitField('Log In')

class PredictionForm(FlaskForm):
    pregnancies = IntegerField('Pregnancies', validators=[
        DataRequired(),
        NumberRange(min=0, max=20, message="Value must be between 0 and 20")
    ])
    
    glucose = FloatField('Glucose (mg/dL)', validators=[
        DataRequired(),
        NumberRange(min=0, max=500, message="Value must be between 0 and 500")
    ])
    
    blood_pressure = FloatField('Blood Pressure (mm Hg)', validators=[
        DataRequired(),
        NumberRange(min=0, max=200, message="Value must be between 0 and 200")
    ])
    
    skin_thickness = FloatField('Skin Thickness (mm)', validators=[
        DataRequired(),
        NumberRange(min=0, max=100, message="Value must be between 0 and 100")
    ])
    
    insulin = FloatField('Insulin (mu U/ml)', validators=[
        DataRequired(),
        NumberRange(min=0, max=1000, message="Value must be between 0 and 1000")
    ])
    
    bmi = FloatField('BMI (weight in kg/(height in m)²)', validators=[
        DataRequired(),
        NumberRange(min=0, max=100, message="Value must be between 0 and 100")
    ])
    
    diabetes_pedigree_function = FloatField('Diabetes Pedigree Function', validators=[
        DataRequired(),
        NumberRange(min=0, max=3, message="Value must be between 0 and 3")
    ])
    
    age = IntegerField('Age (years)', validators=[
        DataRequired(),
        NumberRange(min=0, max=120, message="Value must be between 0 and 120")
    ])
    
    submit = SubmitField('Predict')
