from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODELS_FOLDER'] = 'static/models'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Chargement du dataset diabetes intégré
def load_diabetes_data():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = (diabetes.target > np.median(diabetes.target)).astype(int)  # Conversion en problème de classification
    return X, y

# Fonction pour vérifier l'extension du fichier
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Téléchargement des données
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        flash('Fichier téléchargé avec succès!', 'success')
        return render_template('index.html', filename=filename, default_data=False)
    
    flash('Seuls les fichiers CSV sont autorisés', 'error')
    return redirect(url_for('index'))

# Entraînement du modèle
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Utiliser les données par défaut si demandé
        if 'use_default' in request.form:
            X, y = load_diabetes_data()
            default_data = True
        else:
            filename = request.form['filename']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data = pd.read_csv(filepath)
            
            if data.shape[1] < 2:
                flash('Le fichier doit contenir au moins 2 colonnes (features + target)', 'error')
                return redirect(url_for('index'))
            
            X = data.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
            y = data.iloc[:, -1]   # Dernière colonne
            default_data = False
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Sélection du modèle
        model_name = request.form['model']
        if model_name == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == 'svm':
            model = SVC(probability=True, kernel='linear')
        else:
            flash('Modèle non reconnu', 'error')
            return redirect(url_for('index'))
        
        # Entraînement et évaluation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Génération des visualisations
        plt.switch_backend('Agg')
        
        # Matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non Diabétique', 'Diabétique'],
                   yticklabels=['Non Diabétique', 'Diabétique'])
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        img_confusion = BytesIO()
        plt.savefig(img_confusion, format='png', bbox_inches='tight')
        plt.close()
        confusion_url = base64.b64encode(img_confusion.getvalue()).decode('utf-8')
        
        # Courbe ROC (si le modèle supporte les probabilités)
        roc_url = None
        if hasattr(model, "predict_proba"):
            from sklearn.metrics import RocCurveDisplay
            plt.figure(figsize=(8, 6))
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            plt.title('Courbe ROC')
            img_roc = BytesIO()
            plt.savefig(img_roc, format='png', bbox_inches='tight')
            plt.close()
            roc_url = base64.b64encode(img_roc.getvalue()).decode('utf-8')
        
        return render_template('index.html',
                            accuracy=f"{accuracy:.2%}",
                            report=report,
                            confusion_img=confusion_url,
                            roc_img=roc_url,
                            model_name=model_name.replace('_', ' ').title(),
                            show_results=True,
                            default_data=default_data)
    
    except Exception as e:
        flash(f"Erreur lors de l'entraînement: {str(e)}", "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)