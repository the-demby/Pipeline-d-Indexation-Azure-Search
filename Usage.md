# Guide d'utilisation — Pipeline d'Indexation Azure Search

> Ce document explique étape par étape comment prendre en main et utiliser le script d'indexation Azure Search développé dans ce projet. Il s'adresse aussi bien à un public technique (data, dev) qu'aux utilisateurs avancés souhaitant indexer leurs propres jeux de données sur Azure Search, avec ou sans vectorisation (embeddings).

---

## 1️⃣ Introduction et principes généraux

Ce script a été pensé pour **faciliter l’indexation automatisée** de n’importe quel corpus documentaire (CSV, TXT, etc.) dans Azure Cognitive Search, en supportant l’enrichissement automatique, le découpage optimisé par tokens et l’intégration optionnelle des embeddings Azure OpenAI.

Vous pouvez l’utiliser pour :

* Indexer des bases de documents internes (rapports, extraits de bases, réunions, archives…)
* Alimenter un moteur de recherche intelligent (RAG, semantic search…)
* Préparer des datasets pour l’expérimentation IA sur Azure
* Industrialiser un flux de “Data to Search” custom pour votre entreprise

Le pipeline est découpé en étapes claires : préparation des fichiers → nettoyage/extraction → chunking → embeddings (optionnel) → upload batch Azure Search.

---

## 2️⃣ Préparer son environnement

### 2.1. Prérequis logiciels

* **Python 3.8+**
* Les packages indiqués dans `requirements.txt` (voir README)

### 2.2. Récupérer et paramétrer le projet

* Cloner ou télécharger le repo
* Copier `.env.example` en `.env` et remplir vos secrets Azure / OpenAI

```
AZURE_OPENAI_ED=...    # endpoint Azure OpenAI
AZURE_OPENAI_KEY=...
AZURE_SEARCH_ED=...    # endpoint Azure Search
AZURE_SEARCH_KEY=...
INDEX_NAME=...         # nom personnalisé de votre index Azure
CHUNK_TOK=512          # chunk size tokens (512 recommandé, max 8191 pour ada-002)
EMBED_BEFORE_UPLOAD=False  # ou True si vectorisation locale
```

> **Astuce :** pour traiter des très longs documents (PDF, gros TXT), augmentez la valeur de `CHUNK_TOK` dans la limite du modèle embedding (8191 tokens pour ada-002).

### 2.3. Préparer les données à indexer

* Placez vos fichiers dans le dossier du script ou indiquez le chemin via `Config.DIR` (par défaut, tous les .csv du dossier).
* Pour enrichir les documents avec des données externes (ex : actionnaires), placez le CSV référentiel (`ACTIONNAIRES.csv`) dans le même dossier ou adaptez le chemin.

---

## 3️⃣ Déroulement d'une indexation (pipeline détaillé)

### 3.1. Lecture et préparation des documents

* Le script détecte automatiquement tous les fichiers du type souhaité (`.csv`, `.txt` etc.) selon le paramétrage.
* **CSV** : chaque ligne = un document à indexer. Les champs à prendre comme contenu principal et comme métadonnées sont configurables.
* **TXT** : chaque fichier = un document. Si un fichier dépasse la taille `CHUNK_TOK`, il est automatiquement découpé en plusieurs chunks.
* **Autres formats** : le pipeline est conçu pour pouvoir ajouter facilement de nouveaux parseurs (PDF, DOCX, etc.).

### 3.2. Nettoyage et enrichissement automatique

* Les contenus sont nettoyés (suppression HTML, normalisation espaces/caractères).
* Pour les CSV, si la colonne `name` correspond à une entrée dans `ACTIONNAIRES.csv`, la liste d’actionnaires est ajoutée en tant que métadonnée.
* Les champs sont enrichis ou laissés vides selon les informations disponibles.

### 3.3. Chunking optimisé par tokens

* Tous les documents sont découpés selon le nombre de tokens choisi (512 par défaut, max 8191 pour ada-002).
* Le script s’assure que chaque chunk ne dépasse pas la limite de taille des API Azure (32 766 bytes pour un doc Azure Search).
* Les chunks trop petits (< 200 tokens) sont ignorés (paramètre `MIN_CHUNK_TOK`).

### 3.4. Embeddings Azure OpenAI (optionnel)

* Si `EMBED_BEFORE_UPLOAD=True`, chaque chunk reçoit un embedding Azure OpenAI local avant upload.
* Sinon, la vectorisation peut être réalisée côté Azure, ou bien l’index sera full-text classique.
* Les embeddings sont batchés par 430 docs par défaut pour respecter les limites API Azure.

### 3.5. Upload et suivi de progression

* Les chunks (avec ou sans embeddings) sont uploadés en batchs (par 975 par défaut) sur Azure Search.
* Le script gère les erreurs et retries automatiquement (backoff exponentiel).
* La progression est affichée étape par étape dans le terminal (nombre de fichiers, chunks, batchs uploadés).

---

## 4️⃣ Personnaliser ou étendre le pipeline

### 4.1. Ajouter un nouveau format de fichier (PDF, DOCX, etc.)

* Ajouter une méthode `read_pdf()` ou `read_docx()` dans la classe `DocumentProcessor`.
* Ajouter l’extension dans `SUPPORTED_EXTENSIONS` de la `Config`.
* Plugger la méthode dans le dispatcher `read_documents()`.
* Les étapes de chunking/nettoyage restent valables pour tous formats.

### 4.2. Ajouter un enrichissement spécifique

* Répliquer le pattern du mapping “actionnaires” dans `DocumentProcessor`.
* Exemple : pour ajouter une métadonnée “catégorie” ou “statut”, charger le mapping et l’appliquer lors du passage sur chaque ligne/doc.

### 4.3. Adapter le schéma d’index Azure

* Modifier la méthode `create_or_reset_index` dans `AzureIndexer` pour ajuster les champs indexés, leur type, ou leurs propriétés (filterable, retrievable…).

---

## 5️⃣ Questions fréquentes et conseils

### Peut-on indexer de très gros documents ?

Oui, ils seront automatiquement découpés par tokens selon `CHUNK_TOK` (jusqu’à 8 191 pour ada-002). Le découpage garantit la compatibilité API et la pertinence pour la recherche vectorielle.

### Peut-on désactiver l’enrichissement (ex : actionnaires) ?

Oui, il suffit de retirer ou renommer le fichier référentiel (ex : `ACTIONNAIRES.csv`) ou de désactiver l’appel dans `DocumentProcessor`.

### Comment gérer plusieurs types de documents dans le même pipeline ?

Placez vos fichiers de différents formats dans le dossier cible, ajustez `SUPPORTED_EXTENSIONS`, ajoutez si besoin les parseurs custom, et lancez le script.

### Comment adapter le chunking ou l’upload à mon modèle/usage ?

Changez la valeur de `CHUNK_TOK` dans le `.env` ou directement dans la `Config`. Pour des usages “retrieval only”, vous pouvez utiliser de plus gros chunks.

---

## 6️⃣ Bonnes pratiques

* Testez d’abord sur un petit jeu de fichiers avant un gros batch
* Vérifiez la complétude et la pertinence des champs enrichis (actionnaires, liens, catégories...)
* Utilisez le logging terminal pour détecter les éventuels problèmes d’upload ou d’API
* Sauvegardez/backuppez votre index Azure avant de le supprimer ou de le recréer en prod
* Si vous souhaitez modifier l’algorithme de chunking, ne dépassez jamais la limite du modèle embedding cible

---

## 7️⃣ Support et contribution

* Pour toute question technique, consultez le README du repo et la documentation officielle Azure.
* Pour toute extension métier (nouveau mapping, enrichissement), vous pouvez demander de l’aide ou proposer un “pattern” à la communauté interne.
* Pour contribuer, ouvrez une Pull Request ou documentez vos besoins dans un fichier dédié.

---

Ce guide est évolutif : toute suggestion, retour d’expérience ou demande de clarification est la bienvenue pour améliorer l’usage et l’accessibilité du pipeline d’indexation Azure.