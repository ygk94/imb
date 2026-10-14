---
name: experiment
description: Tester une idée de modèle avec une validation croisée groupée par patient
metadata:
  user-invocable: true
  disable-model-invocation: true
  argument-hint: <idée-du-modèle>
---

Teste l'idée suivante : $1

- Crée ou modifie un script dans `src/` (pas de notebook).
- Validation : GroupKFold (5 plis) sur `patient_id`, métrique RMSE.
- Affiche le RMSE moyen ± écart-type et compare-le au meilleur score précédent s'il est noté dans `src/RESULTS.md`.
- Ajoute une ligne dans `src/RESULTS.md` : date, idée, RMSE CV.
- Ne génère pas de soumission (c'est le rôle de /kaggle-submit).
