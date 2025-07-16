# Azure Search Document Indexer

Ce projet fournit un script Python robuste et généralisable permettant d'indexer efficacement des documents dans un service Azure Cognitive Search, avec ou sans embeddings Azure OpenAI. Il est conçu pour être facilement extensible à de nouveaux types de documents (CSV, TXT, PDF, etc.) et pour faciliter le traitement de corpus volumineux ou hétérogènes, tout en maintenant performance, traçabilité et clarté du pipeline.

---

## 🚀 Fonctionnalités principales

* **Indexation multi-format** : CSV (ligne à ligne), TXT (1 fichier = 1 doc, splitté si trop long), extensible à PDF, DOCX...
* **Nettoyage intelligent** : extraction de texte brut depuis HTML, gestion des champs pertinents, nettoyage unicode.
* **Enrichissement automatique** : mapping dynamique de métadonnées avancées (ex : "actionnaires" pour chaque société à partir d’un CSV dédié).
* **Chunking par tokens** : découpage automatique des documents en chunks optimisés pour les limites du modèle d’embedding OpenAI/Azure.
* **Embeddings Azure OpenAI** : vectorisation optionnelle des chunks avec gestion des quotas et des batchs.
* **Upload massif & robuste** : batching, retry exponentiel, gestion fine des erreurs, suivi de progression dans le terminal.
* **Paramétrage centralisé** : toutes les constantes et variables de pipeline regroupées dans une seule classe Config.
* **Code prêt à industrialiser** : propreté, typage, structure modulaire, facile à étendre et à maintenir.

---

## 📦 Installation

```bash
# Cloner le repo
git clone <repo-url>
cd <repo-folder>

# Installer les dépendances (idéalement dans un venv)
pip install -r requirements.txt
```

#### 📚 Principales dépendances :

* `pandas`, `tiktoken`, `openai`, `azure-search-documents`, `python-dotenv`, `tqdm`, `beautifulsoup4`

---

## ⚙️ Configuration (env)

Créer un fichier `.env` à la racine du projet (voir exemple `.env.example`) :

```
AZURE_OPENAI_ED=<https://...>
AZURE_OPENAI_KEY=<...>
AZURE_SEARCH_ED=<https://...>
AZURE_SEARCH_KEY=<...>
INDEX_NAME=nom_de_mon_index
CHUNK_TOK=512           # chunk size tokens, voir section plus bas
EMBED_BEFORE_UPLOAD=False
```

* **CHUNK\_TOK** : nombre de tokens par chunk (par défaut : 512, max : 8191 pour text-embedding-ada-002).
* **EMBED\_BEFORE\_UPLOAD** : `True` pour générer les embeddings localement (mode RAG vectoriel), sinon indexation full-text simple.

---

## 🏗️ Pipeline & structure du code

```
config.py        # Classe Config (paramètres centralisés)
main.py          # Script principal
└─> DocumentProcessor
       ├─ read_csv()
       ├─ read_txt()
       └─ prepare_content(), prepare_metadata(), enrichissements
└─> AzureIndexer
       ├─ chunk_documents()
       ├─ add_embeddings()
       ├─ create_or_reset_index()
       └─ upload_to_index(), pipeline()
```

* **Facile à étendre** : ajouter simplement une méthode `read_pdf` ou `read_docx` dans `DocumentProcessor` pour supporter de nouveaux formats.
* **Généralisation** : tous les documents sont transformés en "chunks" de texte, quel que soit le format source, avant indexation Azure.

---

## 🔢 Logique de chunking & embeddings

* Tous les documents sont découpés en morceaux (chunks) de taille `CHUNK_TOK` tokens.
* Pour le modèle `text-embedding-ada-002`, la limite maximale est **8191 tokens** (tu peux mettre 8000 pour marge).
* **Par défaut**, 512 tokens pour une meilleure compatibilité cross-modèles.
* Un chunk trop petit (< 200 tokens, paramètre : MIN\_CHUNK\_TOK) n'est pas indexé.
* Si EMBED\_BEFORE\_UPLOAD=True, chaque chunk reçoit son embedding avant upload ; sinon c'est Azure qui gère la vectorisation côté cloud.

---

## 📂 Organisation des données à indexer

* **CSV** : chaque ligne (hors "actionnaires.csv") devient un document.

  * Enrichissement dynamique avec le mapping "actionnaires" si la colonne "name" correspond.
* **TXT** : chaque fichier texte devient un document (splitté si trop long pour un chunk).
* **PDF (bientôt)** : ajouter une méthode `read_pdf()` dans `DocumentProcessor` pour traiter les PDF.

---

## 🔑 Champs d'index Azure générés

* `id` (clé unique du chunk)
* `content` (texte chunké, nettoyé)
* `embedding` (vecteur embedding, si applicable)
* `search_result_link`, `page_language`, `actionnaires`, `filename` (métadonnées)

Le mapping et le schéma de l’index sont généralisables et adaptables à vos besoins spécifiques.

---

## 🚦 Exécution rapide

```bash
python main.py
```

* Suit la progression dans le terminal.
* Uploads robustes avec batch/retry.
* Logique d’enrichissement (ex: actionnaires) automatique si CSV fourni.
* Output final : index Azure prêt à la recherche full-text, sémantique ou vectorielle.

---

## 📈 Extensibilité / Bonnes pratiques

* **Pour ajouter un nouveau format** : implémenter une méthode dédiée (ex: `read_pdf`) dans `DocumentProcessor`, et ajouter l’extension dans `Config.SUPPORTED_EXTENSIONS`.
* **Pour de nouveaux enrichissements** : répliquer le pattern de mapping (ex: actionnaires) dans le processor.
* **Pour changer les champs indexés** : éditer la méthode `create_or_reset_index` dans `AzureIndexer`.

---

## 🤝 Contributions

* Merci de proposer vos améliorations via Pull Request ou Issue.
* Pour toute question sur l’usage ou l’extension, consulter la documentation complète (à venir).

---

## 📝 Auteurs

* Script d’indexation : \[votre nom ou organisation]
* Contact : \[email/contact]

---

## 📚 Voir aussi

* [Docs Azure Cognitive Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search)
* [Docs Azure OpenAI Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/embeddings)
* [OpenAI Embedding Models](https://platform.openai.com/docs/guides/embeddings)

---

Ce script est fourni “as is” : il a été testé pour de nombreux volumes et formats, mais la personnalisation reste à la main de chaque équipe pour adapter au contexte métier ou aux nouvelles exigences techniques.
