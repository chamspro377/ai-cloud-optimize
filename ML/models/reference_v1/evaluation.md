# Premier modèle Alibaba — évaluation

Exécution UTC : 2026-09-14T09:44:07.657152+00:00

Deux forêts aléatoires multi-sorties (CPU et RAM), comparées à une constante
égale à la moyenne des cibles d'apprentissage. Paramètres fixés avant le test.

Séparation commune : 9314 jobs d'apprentissage et 2329 jobs de test. Aucun job partagé.

| Cible | Tâches test | MAE constante | MAE forêt | Réduction MAE | R² forêt |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU moyen par tentative | 13628 | 0.307513 | 0.247130 | 19.6% | 0.273 |
| Pic CPU observé | 13628 | 1.232477 | 0.980366 | 20.5% | 0.311 |
| RAM moyenne par tentative | 6430 | 0.009648 | 0.008771 | 9.1% | 0.129 |
| Pic RAM observé | 6430 | 0.018764 | 0.012165 | 35.2% | 0.487 |

MAE : erreur absolue moyenne, dans l'unité de la cible. Une valeur plus basse est meilleure.
La réduction MAE est relative à la constante, pas un pourcentage de précision.
R² : part de variabilité expliquée sur le test ; ce score peut être négatif.

## Portée des résultats

Le détail des erreurs d'apprentissage, RMSE, sous-estimations et couvertures
complètes/partielles se trouve dans `evaluation.json` et le notebook 03.
Les fichiers de prédictions permettent de recalculer les scores. Les modèles
sauvegardés sont exactement ceux évalués, sans réentraînement sur le test.

Les scores indiquent un signal prédictif encore limité. Aucun seuil de qualité
opérationnelle n'est établi ; une amélioration face à une constante ne suffit
pas à valider le dimensionnement automatique des VM.

- Random held-out jobs measure generalization within this trace, not future-time or VM performance.
- CPU stays in original source units; memory stays normalized. Predictions are not vCPU/GB recommendations.
- Targets summarize measured completed attempts; missing measurements and completed-only selection may bias results.
- High coverage only means coverage among accepted completed attempts, not all requested instances.
- Observed peaks are not simultaneous total task consumption and are not guaranteed upper bounds.
- These three Alibaba request features do not map directly to the current API's expected_users/workload_type.
- Test results are now visible; further model selection requires training-only validation and a fresh final evaluation protocol.

## Reproduire

Depuis la racine du projet :

```powershell
.venv\Scripts\python.exe -m ML.src.train
.venv\Scripts\python.exe -m ML.src.predict --instance-num 2 --plan-cpu 50 --plan-mem 0.005
```

Les trois entrées proviennent du schéma Alibaba. Elles ne correspondent pas
encore aux entrées métier `expected_users` ou `workload_type` de l'API.

## Références

- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/schema.csv
