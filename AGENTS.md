# Projet : prédiction du score OFF (Kaggle, Parkinson)

## Données (racine, ne jamais lire directement : trop lourdes)
- `X_train.csv`, `y_train.csv` (colonnes `Index,target`), `X_test.csv`, `sample_submission.csv`
- Une ligne = une visite. Colonnes : patient_id, cohort, sexM, gene, age_at_diagnosis, age, ledd,
  time_since_intake_on, time_since_intake_off, rater_id, on, off. Beaucoup de NaN.
- Cible : score OFF à chaque visite. Métrique : RMSE (plus bas = mieux).
- Pour explorer : écrire/lancer un script qui affiche un résumé (shape, dtypes, NaN, describe).

## Environnement
- Windows. Python : `.venv\Scripts\python.exe` (ne pas créer d'autre venv).
- Paquets : pandas, numpy, scikit-learn, skrub, skore, lightgbm, xgboost, catboost, matplotlib, seaborn.

## Structure
- `src/` : code (scripts .py)
- `submissions/` : CSV de soumission générés
- `plans/` : plans écrits en mode Plan
