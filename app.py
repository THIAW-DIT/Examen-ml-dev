from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'

# Page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Téléchargement des données
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(url_for('index'))
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        flash('Fichier téléchargé avec succès')
        return redirect(url_for('index'))

# Entraînement du modèle
@app.route('/train', methods=['POST'])
def train_model():
    # Récupérer les données
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], request.form['filename'])
    data = pd.read_csv(filepath)
    
    # Sélection des caractéristiques et de la cible
    X = data.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
    y = data.iloc[:, -1]   # Dernière colonne
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Sélection du modèle
    model_name = request.form['model']
    if model_name == 'linear_regression':
        model = LinearRegression()
    elif model_name == 'random_forest':
        model = RandomForestClassifier()
    elif model_name == 'svm':
        model = SVC()
    else:
        flash('Modèle non reconnu')
        return redirect(url_for('index'))
    
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédiction et évaluation
    y_pred = model.predict(X_test)
    if model_name == 'linear_regression':
        mse = mean_squared_error(y_test, y_pred)
        result = f'Erreur quadratique moyenne (MSE): {mse}'
    else:
        accuracy = accuracy_score(y_test, y_pred)
        result = f'Précision: {accuracy * 100:.2f}%'
    
    # Sauvegarder le modèle
    modelpath = os.path.join(app.config['MODELS_FOLDER'], f'{model_name}.pkl')
    pd.to_pickle(model, modelpath)
    
    # Générer un graphique
    plt.figure()
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plotpath = os.path.join('static', 'plot.png')
    plt.savefig(plotpath)
    
    return render_template('index.html', result=result, plot_url=plotpath)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['MODELS_FOLDER']):
        os.makedirs(app.config['MODELS_FOLDER'])
    app.run(debug=True)