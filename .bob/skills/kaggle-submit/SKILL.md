---
name: kaggle-submit
description: Entraîne le modèle choisi, crée et publie son EstimatorReport skore sur le Skore Hub, puis écrit le CSV de soumission Kaggle (Index,target) et donne l'URL du rapport à mettre dans la description.
---

Suivre ces étapes dans l'ordre :

1. Identifier le script/modèle à soumettre (demander à l'utilisateur si ce n'est pas clair).
2. Vérifier le score de validation croisée groupée par `patient_id` (GroupKFold) et l'afficher (RMSE moyen ± écart-type).
3. Créer un `skore.EstimatorReport` du modèle (split train/test groupé par patient) et le publier :
   `project = skore.Project("<nom-du-projet>", mode="hub", workspace="<workspace-hub>")` puis `project.put("<clé>", report)`.
   Si le workspace ou le nom du projet Hub est inconnu, le demander. Si l'utilisateur n'est pas authentifié, lui dire de lancer lui-même `.venv\Scripts\skore.exe hub api-key generate --workspace <workspace-hub>`.
4. Réentraîner sur tout X_train/y_train, prédire X_test.
5. Écrire `submissions/<date>_<nom-modèle>.csv` avec les colonnes `Index,target`, en vérifiant :
   même nombre de lignes et mêmes Index que sample_submission.csv, aucune valeur NaN.
6. Terminer par un récapitulatif : RMSE CV, chemin du CSV, URL du rapport Skore Hub à coller dans la description de la soumission Kaggle.
