# Préparation des données ML

## État du projet

Le backend fournit encore des recommandations par règles. Les données Alibaba
2017 sont préparées et deux premiers modèles CPU/RAM sont entraînés et évalués.
Ils ne sont pas encore branchés sur l'API.

## Reproduire

Depuis la racine de `ai-cloud-optimizer`, avec l'environnement Python du projet :

```powershell
.venv\Scripts\python.exe -m ML.src.build_dataset
.venv\Scripts\python.exe -m unittest discover -s ML/tests -v
```

Le script accepte `--raw-dir`, `--output-dir` et `--chunksize` (100 000 par défaut).
Il vérifie d'abord les 12 champs de chaque enregistrement d'instance, puis relit
le fichier par blocs typés. La mémoire utilisée dépend du bloc et du nombre de
tâches, pas du nombre total d'instances. Les fichiers bruts ne sont pas modifiés.

Entrées : `ML/data/raw/batch_task.csv` et `batch_instance.csv`.
Sorties régénérables :

- `ML/data/processed/task_resource_dataset.csv` : une ligne par tâche admissible.
- `ML/data/processed/data_quality_report.json` : compteurs, provenance et contrat du modèle.

Le notebook `02_data_preparation.ipynb` lit ces sorties et explique les contrôles.
Il ne relance pas automatiquement le traitement du gros fichier.

## Règles de préparation

La clé `(job_id, task_id)` doit être unique dans les tâches, sinon le traitement
s'arrête. Les tentatives doivent être terminées, avoir une clé présente et
positive qui correspond à une tâche, et une fin positive postérieure ou égale
au début. Un début négatif reste accepté. Les exclusions sont comptées selon
cet ordre, sans double comptage dans le bilan principal.

Pour CPU et RAM séparément, moyenne et maximum doivent être présents, finis,
non négatifs et ordonnés. Les diagnostics de ressources peuvent se chevaucher.
Une mesure manquante n'est jamais imputée à zéro ; les vrais zéros sont gardés.

Seules les tâches terminées avec `instance_num`, `plan_cpu` et `plan_mem`
strictement positifs et au moins une cible utilisable sont exportées.

## Dictionnaire et contrat d'apprentissage

| Colonnes | Rôle |
| --- | --- |
| `job_id`, `task_id` | Clés, jamais des prédicteurs ; `job_id` sert à séparer les jeux |
| `instance_num`, `plan_cpu`, `plan_mem` | Caractéristiques disponibles à la demande |
| `create_timestamp`, `end_timestamp`, `status` | Contexte seulement, hors prédicteurs |
| `completed_attempt_count` | Nombre de tentatives terminées acceptées |
| `cpu/mem_measured_attempt_count` | Nombre de paires de mesures valides |
| `cpu/mem_mean_of_attempt_averages` | Moyenne des moyennes, pondérée par le nombre de tentatives mesurées |
| `cpu/mem_observed_peak` | Maximum des pics observés ; pas une somme de consommations simultanées |
| `cpu/mem_measurement_coverage` | Mesures valides / tentatives terminées acceptées |
| `eligible_cpu`, `eligible_mem` | Filtre obligatoire pour sélectionner la cible correspondante |

Les préfixes `cpu/mem` indiquent deux colonnes distinctes dans le CSV.
Les moyennes ne sont pas pondérées par la durée. Le calcul additionne sommes et
effectifs à travers les blocs ; il ne fait pas une moyenne de moyennes de blocs.
Les pics, couvertures et compteurs ne doivent pas devenir des prédicteurs.

Séparer par job avant tout entraînement ou ajustement de transformation.
Les cibles CPU/RAM peuvent provenir de sous-ensembles différents de tentatives.
La couverture est relative aux tentatives acceptées et non à `instance_num`.
Le fichier ne possède pas de clé d'instance unique permettant de dédupliquer
fiablement les tentatives : aucune suppression arbitraire n'est effectuée.

## Limites

La mémoire est normalisée et n'est pas une valeur en Go. Les valeurs CPU restent
dans les unités du fichier ; aucun ratio demande/usage, conversion en vCPU ou
division par 100 n'est appliqué. La correspondance entre ces mesures et les
capacités des VM devra être établie avant un dimensionnement opérationnel.

Les mesures manquantes, les clés absentes et la sélection des tâches terminées
créent un risque de biais. Un pic observé sur quelques tentatives peut sous-estimer
le pic réel. Un modèle entraîné ici n'explique pas encore les entrées
`expected_users` ou `workload_type` de l'API actuelle.

Sources :
- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/schema.csv
- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md

## Premier entraînement et prédiction

```powershell
.venv\Scripts\python.exe -m ML.src.train
.venv\Scripts\python.exe -m ML.src.predict --instance-num 2 --plan-cpu 50 --plan-mem 0.005
```

Les sorties sont dans `ML/models/reference_v1/` : modèles `cpu_reference.joblib`
et `mem_reference.joblib`, scores `evaluation.json`, synthèse `evaluation.md`,
séparation exacte `job_split.csv` et prédictions des deux jeux de test.
Le notebook `03_model_baseline.ipynb` permet de consulter et vérifier les résultats.

Deux forêts aléatoires multi-sorties utilisent des paramètres fixés à l'avance :
100 arbres, profondeur maximale 12, 5 observations minimum par feuille, graine 42.
Une constante égale à la moyenne des cibles d'apprentissage sert de comparaison.
80 % des jobs sont réservés à l'apprentissage, 20 % au test, avec une partition
commune avant le filtrage CPU/RAM. Les proportions de lignes peuvent différer.

Les prédicteurs sont exclusivement `instance_num`, `plan_cpu`, `plan_mem`.
Les modèles sauvegardés sont ceux évalués ; ils ne sont pas réentraînés sur
l'ensemble des données. Le test ne sert ni à régler les paramètres ni à choisir
le modèle. Les métriques comprennent MAE, RMSE, R², taux de sous-estimation et
scores par couverture des mesures. Le rapport enregistre les versions Python
et bibliothèques, les paramètres et le SHA-256 du CSV utilisé.

Le traitement utilise un seul worker par défaut, compatible avec l'environnement
Windows restreint. `--n-jobs` permet de l'adapter hors de cet environnement.
Les modèles `.joblib` sont générés localement ; charger seulement des modèles
de provenance connue. La fonction de prédiction signale les entrées hors des
plages d'apprentissage, sans en déduire une garantie de fiabilité pour les autres.

Cette première évaluation mesure la généralisation à d'autres jobs de la même
trace. Elle ne valide pas les performances sur des VM ni à une période future.
Le réglage ultérieur devra utiliser une validation sur les jobs d'apprentissage,
et une évaluation externe ou temporelle sera nécessaire avant usage opérationnel.
