# Règles du projet Kaggle

- Validation : toujours `GroupKFold` (ou équivalent groupé) sur `patient_id`. Le leaderboard sépare les patients.
- Ne jamais utiliser `patient_id` comme feature.
- Toute soumission = CSV `Index,target`, une ligne par visite de X_test, même ordre/format que sample_submission.csv.
- Phase actuelle = développement : utiliser skore en local uniquement (`skore.Project("kaggle-off", mode="local")`), jamais le Hub sans demande explicite.
- Au moment d'une soumission (seulement) : un `skore.EstimatorReport` publié sur le Skore Hub du lab ; donner son URL à l'utilisateur pour la description Kaggle.
- Ne jamais lire les CSV en entier ni les mentionner avec @ ; passer par un script de résumé.
- Ne jamais demander, afficher ou écrire de token/identifiant ; l'utilisateur s'authentifie lui-même au Hub (`skore.exe hub api-key generate`).
- Commits git : messages simples, sans aucune ligne de co-auteur IA.
- Réponses concises, en français.
