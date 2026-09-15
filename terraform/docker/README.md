# Démonstration Terraform + Docker

Ce dossier est la cible locale du projet. Exécuter les commandes **dans Ubuntu**,
où Docker et Terraform doivent être installés. Le compte doit déjà avoir accès
au daemon Docker local (`docker info`). Le provider utilise son socket Unix.

L'application tourne séparément avec Compose sur le port 8001.
Terraform gère uniquement `ai-cloud-demo`, un serveur Nginx sur le port 8080.
Il ne crée pas de VM et ne transforme pas les prédictions Alibaba en ressources.

| Profil | Limite CPU | Mémoire (MiB) |
| --- | ---: | ---: |
| small (défaut) | 0,5 | 256 |
| medium | 1 | 512 |
| large | 2 | 1024 |

Ces profils sont des limites de démonstration, pas des ressources réservées
ni des recommandations ML validées. Aucune limite disque ou HA n'est configurée.
L'image `nginx:stable-alpine` nécessite Internet au premier téléchargement ;
le tag peut évoluer et n'est pas verrouillé par digest.

## Créer et vérifier

Depuis la racine du dépôt :

```sh
terraform -chdir=terraform/docker init
terraform -chdir=terraform/docker fmt -check
terraform -chdir=terraform/docker validate
terraform -chdir=terraform/docker plan -var="profile=small" -out=demo.tfplan
terraform -chdir=terraform/docker apply demo.tfplan
curl --fail http://127.0.0.1:8080
docker inspect ai-cloud-demo --format '{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}'
```

Examiner le plan avant l'application. Pour small, les limites attendues sont
`500000000` NanoCPUs et `268435456` octets. Le site affiche la page Nginx.
Ne pas lancer les commandes depuis `terraform/` seul : ce dossier parent
contient l'ancien exemple ESXi. Ses fichiers de variables ne sont pas chargés
par `terraform/docker`. Ne pas y copier l'export VM `sizing.auto.tfvars.json`.

## Changer de profil

```sh
terraform -chdir=terraform/docker plan -var="profile=medium" -out=demo.tfplan
terraform -chdir=terraform/docker apply demo.tfplan
```

Terraform peut recréer le conteneur pour changer ses limites. Le profil choisi
doit être fourni aux prochains plans (sans variable, le défaut est small).

## Accès depuis Windows

```sh
ssh -L 8001:127.0.0.1:8001 -L 8080:127.0.0.1:8080 utilisateur@IP_VM
```

Adapter compte/IP, puis ouvrir http://127.0.0.1:8001 (application) et
http://127.0.0.1:8080 (démo). Remplacer l'ancien tunnel s'il occupe déjà 8001.

## Conserver les résultats et nettoyer

Après vérification, conserver la configuration effective pour le rapport :

```sh
mkdir -p results
terraform -chdir=terraform/docker output -json > results/terraform-docker.json
docker inspect ai-cloud-demo --format '{{json .HostConfig}}' > results/demo-limits.json
terraform -chdir=terraform/docker plan -destroy -out=destroy.tfplan
terraform -chdir=terraform/docker apply destroy.tfplan
docker compose exec app python scripts/check_app.py
```

La destruction supprime uniquement la démo gérée par ce dossier ; Compose
continue de gérer l'application. L'image téléchargée reste en cache.
Ne pas publier les états ou plans Terraform. Conserver le fichier de verrouillage
`.terraform.lock.hcl` généré par init pour fixer la version du provider.

État : configuration préparée ; `init`, `validate`, `plan` et le déploiement
réel restent à exécuter dans Ubuntu, les outils étant absents du PC actuel.

Référence : [provider Docker](https://registry.terraform.io/providers/kreuzwerker/docker/3.6.1/docs/resources/container).
