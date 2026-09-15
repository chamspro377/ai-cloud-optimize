# AI Cloud Optimizer

Prototype local : recommandations VM par règles, prédictions Alibaba par Random
Forest et export des paramètres Terraform. Aucun entraînement nécessaire.

## Lancer sur Windows

Depuis la racine du projet :

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Ouvrir http://127.0.0.1:8000 ; documentation API : http://127.0.0.1:8000/docs.
Utiliser l'environnement existant, qui contient déjà les dépendances et modèles.

## Lancer avec Docker (PC ou VM Ubuntu)

Prérequis : Docker avec le plugin Compose, en mode conteneurs Linux.
Depuis la racine du dépôt :

```sh
docker compose up -d --build
docker compose ps
docker compose exec app python scripts/check_app.py
```

Ouvrir http://127.0.0.1:8001. Le port 8001 évite le conflit avec le serveur
Python existant sur 8000. L'image utilise Python 3.14 et les mêmes versions ML
que les modèles enregistrés. La construction vérifie une vraie prédiction :
elle échoue si les modèles sont incompatibles. Aucun réentraînement.
Le téléchargement des dépendances nécessite Internet à la première construction.
Si une version ou une roue Linux manque, conserver les versions ML et résoudre
le problème de paquet avant de poursuivre.

Seuls le code nécessaire, l'interface et les deux modèles sont copiés.
Les CSV Alibaba, `.venv`, Terraform et les secrets sont exclus du contexte.
Le processus utilise un utilisateur non root. Le contrôle de santé vérifie
l'API et la présence des modèles ; la commande de vérification ci-dessus teste
aussi les prédictions et l'export.

```sh
docker compose logs --tail=50 app
docker compose down
```

Pour accéder à l'application dans la VM depuis le navigateur du laptop,
conserver l'écoute locale et ouvrir un tunnel depuis Windows (adapter le compte
et l'IP de la VM) :

```sh
ssh -L 8001:127.0.0.1:8001 utilisateur@IP_VM
```

Puis ouvrir http://127.0.0.1:8001 sur Windows en gardant le tunnel ouvert.
Par défaut, Compose publie le port uniquement sur localhost, conformément à la
[documentation Docker](https://docs.docker.com/engine/network/port-publishing/).

État : fichiers préparés ; construction et exécution Docker à vérifier sur une
machine équipée de Docker. Le conteneur de démonstration est configuré séparément dans
`terraform/docker` ; l'export actuel reste destiné aux VM.

## Démonstration

1. Calculer une configuration pour 100 utilisateurs : 2 vCPU, 4 Go, 50 Go, une VM.
2. Passer à 2500 utilisateurs en production : 8 vCPU, 16 Go, 250 Go par VM.
3. Tester le ML avec `instance_num=2`, `plan_cpu=50`, `plan_mem=0.005`.
4. Télécharger `sizing.auto.tfvars.json` depuis la configuration de VM.

Les règles utilisent les seuils 500 et 2000 utilisateurs. La production ajoute
50 Go de disque. L'option deux VM double le nombre de machines, sans configurer
de haute disponibilité. Le type de charge n'influence pas encore ces règles.
Les résultats ML sont expérimentaux, dans les unités Alibaba ; la RAM n'est pas
en Go. Les modèles attendent des caractéristiques de tâches, pas un nombre
d'utilisateurs. Ces deux fonctions sont donc présentées séparément.

## Terraform : démonstration Docker locale

La cible retenue est une VM Ubuntu locale avec Docker. Le dossier
[`terraform/docker`](terraform/docker/README.md) gère un unique conteneur Nginx
sur le port 8080, avec trois profils CPU/RAM. L'application Compose reste
indépendante sur le port 8001. Suivre son README dans Ubuntu pour créer,
vérifier, conserver les résultats et supprimer la démonstration.

Les profils sont choisis manuellement, indépendamment du ML. L'export de
l'interface reste un export **VM** et ne doit pas être utilisé dans ce dossier.
L'ancien exemple ESXi est conservé dans `terraform/main.tf` à titre historique ;
ne pas exécuter Terraform depuis le dossier parent pour la démo locale.

## Vérifications rapides

```powershell
.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

La comparaison ML avancée a été réalisée séparément ; ce prototype utilise les
modèles `ML/models/reference_v1` déjà présents et testés. Le gagnant de la
comparaison est aussi Random Forest. Les rapports et graphiques sont reportés.

## Récupérer le projet depuis GitHub

```sh
git clone https://github.com/chamspro377/ai-cloud-optimize.git
cd ai-cloud-optimize
docker compose up -d --build
docker compose exec app python scripts/check_app.py
```

Ces commandes nécessitent que cette version soit publiée. Les deux modèles
de référence sont inclus ; les données brutes Alibaba et l'environnement Python
local ne sont pas nécessaires au lancement et sont exclus. Ne charger que les
fichiers joblib de confiance du projet.

Le commit initial contenait `.venv`. Son retrait du suivi Git pour les prochains
commits conserve les fichiers locaux ; l'ancien historique n'est pas réécrit.
