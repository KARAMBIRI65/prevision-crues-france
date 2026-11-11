# Prévision de crues en France

Prévoir si une rivière va déborder dans les 3 prochains jours.

Les données viennent de la base CAMELS-FR de l'INRAE, 19 rivières françaises de 1999 à 2018. Elles sont déjà dans le dossier `data/`.

J'utilise la pluie, la température et le niveau de la rivière. J'ai comparé plusieurs modèles, XGBoost est le meilleur.

Sur 2015 à 2018, avec une prévision de pluie simulée, 79 % des crues sont annoncées à l'avance.

Une donnée par jour, c'est trop peu pour les crues très rapides.

Pour le lancer, installer les dépendances avec `pip install -r requirements.txt` puis ouvrir `notebooks/prevision_crues.ipynb`.

Issa Aziz Karambiri
