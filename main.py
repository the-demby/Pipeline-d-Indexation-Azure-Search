import os
import glob
import json
import ijson
import decimal
import time
import random
from pathlib import Path
import pandas as pd
import tiktoken

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticSearch, SemanticField,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
)
from dotenv import load_dotenv
from tqdm import tqdm
from bs4 import BeautifulSoup

# --------------- CONFIG ---------------

class Config:
    """
    Paramètres du pipeline. Tout est overridable via le .env (cf. doc en haut du script).
    """
    def __init__(self):
        load_dotenv()
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ED")
        self.AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
        self.AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ED")
        self.AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
        self.INDEX_NAME = os.getenv("INDEX_NAME")
        self.DIR = "*.csv"
        self.ACTIONNAIRE_CSV = "data/ACTIONNAIRES.csv"
        self.CHUNK_TOK = int(os.getenv("CHUNK_TOK", "512"))
        self.CHUNK_OVER = 0
        self.MIN_CHUNK_TOK = 50
        self.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        self.EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada")
        self.EMBED_BATCH = int(os.getenv("EMBED_BATCH", "250"))
        self.BATCH_UPLOAD = int(os.getenv("BATCH_UPLOAD", "500"))
        self.JSON_PATH = os.getenv("JSON_PATH", "chunked_docs_final.json")
        self.JSONL_PATH = os.getenv("JSONL_PATH", "chunked_docs_final.jsonl")
        self.ERROR_LOG_PATH = os.getenv("ERROR_LOG_PATH", "failed_uploads.log")
        self.EMBED_BEFORE_UPLOAD = os.getenv("EMBED_BEFORE_UPLOAD", "False").lower() == "true"
        self.UPLOAD_ONLY = os.getenv("UPLOAD_ONLY", "False").lower() == "true"
        self.RECREATE_INDEX = os.getenv("RECREATE_INDEX", "True").lower() == "true"
        self.SAVE_CHUNKS_LOCALLY = os.getenv("SAVE_CHUNKS_LOCALLY", "True").lower() == "true"
        self.UPLOAD_STREAMING = os.getenv("UPLOAD_STREAMING", "False").lower() == "true"
        self.USE_JSONL = os.getenv("USE_JSONL", "False").lower() == "true"
        self.SUPPORTED_EXTENSIONS = [ext.strip() for ext in os.getenv("SUPPORTED_EXTENSIONS", ".csv,.txt").split(",")]
        self.CONTENT_FIELDS = [
            "search_result_snippet", "page_description", "page_content"
        ]
        self.METADATA_FIELDS = [
            "search_result_title", "search_result_link", "page_language"
        ]
        self.SHOW_PROGRESS = True

# --------------- DOCUMENT PROCESSOR ---------------

class DocumentProcessor:
    """
    Lecture/préparation multi-format, AVEC gestion spécifique d'enrichissement Actionnaires (CSV).
    Pour ajouter un nouveau type, étendre la méthode read_<ext> (ex: read_pdf).
    """
    def __init__(self, config: Config):
        self.config = config
        self.actionnaires_dict = self.load_actionnaires(self.config.ACTIONNAIRE_CSV)

    def load_actionnaires(self, actionnaire_csv):
        if not Path(actionnaire_csv).exists():
            return {}
        def extract_actionnaires_from_results(results_str):
            import ast, re
            try:
                lst = ast.literal_eval(results_str)
                noms = []
                for d in lst:
                    snippet = d.get("snippet", "")
                    found = re.findall(r"actionnaire[s]? [:\-\–]?\s?([A-Za-z0-9,.\(\) ]+)", snippet, flags=re.I)
                    for f in found:
                        noms.append(f.strip())
                return "; ".join(set(noms)) if noms else "Non trouvé"
            except Exception:
                return "Non trouvé"
        for encoding in ["utf-8-sig", "utf-16"]:
            try:
                df = pd.read_csv(actionnaire_csv, encoding=encoding)
                break
            except Exception:
                continue
        else:
            print(f"[Erreur] Impossible de lire {actionnaire_csv}")
            return {}
        df["name"] = df["name"].astype(str).str.strip().str.upper()
        df["actionnaires"] = df["results"].apply(extract_actionnaires_from_results)
        return dict(zip(df["name"], df["actionnaires"]))

    def read_documents(self):
        all_docs = []
        files = glob.glob(self.config.DIR)
        for file in files:
            ext = Path(file).suffix.lower()
            if ext == ".csv" and self.config.ACTIONNAIRE_CSV.lower() in file.lower():
                continue
            if ext == ".csv":
                docs = self.read_csv(file)
            elif ext == ".txt":
                docs = self.read_txt(file)
            # elif ext == ".pdf": docs = self.read_pdf(file)
            else:
                continue
            all_docs.extend(docs)
        if self.config.SHOW_PROGRESS:
            print(f"[Processor] {len(all_docs)} documents extraits et préparés.")
        return all_docs

    def read_csv(self, filepath):
        docs = []
        for encoding in ["utf-8-sig", "utf-16"]:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except Exception:
                continue
        else:
            print(f"[Erreur] Impossible de lire {filepath}")
            return docs
        for _, row in df.iterrows():
            content = self.prepare_content(row, self.config.CONTENT_FIELDS)
            metadata = self.prepare_metadata(row, self.config.METADATA_FIELDS, row)
            if content:
                doc = {"content": content, **metadata}
                docs.append(doc)
        return docs

    def read_txt(self, filepath):
        """
        Pour chaque fichier .txt : on récupère tout le texte comme un seul doc (un CR, un rapport, etc.)
        S'il dépasse CHUNK_TOK, il sera automatiquement découpé par la logique de chunking (en aval).
        """
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        doc = {
            "content": content.strip(),
            "filename": os.path.basename(filepath),
        }
        return [doc]

    # Pour pdf, plugger ici une méthode read_pdf qui lit et renvoie [{"content": ..., "filename": ...}, ...]

    def prepare_content(self, row, content_fields):
        content = ""
        for field in content_fields:
            value = row.get(field, "")
            if self.is_valid_str(value):
                content += self.clean_text(value) + "\n"
        return content.strip()

    def prepare_metadata(self, row, metadata_fields, row_full=None):
        metadata = {}
        for field in metadata_fields:
            value = row.get(field, "")
            if self.is_valid_str(value):
                metadata[field] = value
        company_name = row.get("name", None) if row_full is not None else None
        if self.actionnaires_dict and company_name:
            meta_name = str(company_name).strip().upper()
            metadata["actionnaires"] = self.actionnaires_dict.get(meta_name, "Non trouvé")
        else:
            metadata["actionnaires"] = "Non trouvé"
        return metadata

    @staticmethod
    def is_valid_str(value):
        return pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan"

    @staticmethod
    def clean_text(text):
        if not text:
            return ""
        try:
            text = BeautifulSoup(str(text), "html.parser").get_text()
        except Exception:
            text = str(text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = " ".join(text.split())
        return text

# --------------- AZURE INDEXER ---------------

class AzureIndexer:
    """
    Gère la logique Azure Search : création index, chunking, embedding, upload.
    Affichage feedback terminal (progression) uniquement.
    """
    def __init__(self, config: Config):
        self.config = config

    def chunk_documents(self, documents):
        """
        Découpe les docs en chunks de tokens, gère la taille max Azure.
        Pour .txt ou .pdf : chaque fichier → 1 chunk si court, splitté sinon.
        Pour .csv : chaque ligne (déjà un doc) → splitté si > chunk_size.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        chunked_docs = []

        def split_chunk_tokens(tokens, doc, base_id, max_bytes):
            if not tokens:
                return
            chunk_text = encoding.decode(tokens)
            size = len(chunk_text.encode("utf-8"))
            if size <= 32766:
                new_doc = dict(doc)
                new_doc["content"] = chunk_text
                new_doc["id"] = base_id
                chunked_docs.append(new_doc)
            else:
                mid = len(tokens) // 2
                split_chunk_tokens(tokens[:mid], doc, base_id + "_a", max_bytes)
                split_chunk_tokens(tokens[mid:], doc, base_id + "_b", max_bytes)

        for idx, doc in enumerate(documents):
            tokens = encoding.encode(doc["content"])
            if not tokens:
                continue
            for i in range(0, len(tokens), self.config.CHUNK_TOK - self.config.CHUNK_OVER):
                chunk_tokens = tokens[i:i + self.config.CHUNK_TOK]
                if len(chunk_tokens) < self.config.MIN_CHUNK_TOK:
                    continue
                base_id = f"doc_{idx}_{i}"
                split_chunk_tokens(chunk_tokens, doc, base_id, 32766)

        if self.config.SHOW_PROGRESS:
            print(f"[Indexer] {len(chunked_docs)} chunks créés.")
        return chunked_docs

    def add_embeddings(self, chunked_docs):
        oaiclient = AzureOpenAI(
            api_key=self.config.AZURE_OPENAI_KEY,
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
            api_version="2024-10-21"
        )
        print("[Indexer] Génération des embeddings Azure OpenAI...")
        for i in tqdm(range(0, len(chunked_docs), self.config.EMBED_BATCH), desc="Embedding batches"):
            batch_docs = chunked_docs[i:i+self.config.EMBED_BATCH]
            contents = [doc["content"] for doc in batch_docs]

            def embed_call():
                resp = oaiclient.embeddings.create(input=contents, model=self.config.EMBEDDING_DEPLOYMENT_NAME)
                for idxb, doc in enumerate(batch_docs):
                    doc["embedding"] = resp.data[idxb].embedding
            self.exponential_backoff_retry(embed_call, max_retries=6, base_delay=5)
        print("[Indexer] Embeddings générés pour tous les chunks.")
        return chunked_docs

    @staticmethod
    def exponential_backoff_retry(func, max_retries=7, base_delay=2):
        delay = base_delay
        for attempt in range(1, max_retries + 1):
            try:
                return func()
            except Exception as e:
                print(f"[Backoff] Erreur (tentative {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    raise
                sleep_time = delay + random.uniform(0, 0.5 * delay)
                print(f"Pause {sleep_time:.1f}s avant retry...")
                time.sleep(sleep_time)
                delay *= 2

    def save_chunks_to_json(self, docs):
        with open(self.config.JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        print(f"[Indexer] Chunks sauvegardés dans {self.config.JSON_PATH}")

    def save_chunks_to_jsonl(self, docs):
        with open(self.config.JSONL_PATH, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"[Indexer] Chunks sauvegardés (jsonlines) dans {self.config.JSONL_PATH}")

    def create_or_reset_index(self):
        if not self.config.RECREATE_INDEX:
            print("[Indexer] (Re)création d'index désactivée (RECREATE_INDEX=False)")
            return
        idx_client = SearchIndexClient(
            endpoint=self.config.AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(self.config.AZURE_SEARCH_KEY)
        )
        try:
            if self.config.INDEX_NAME in [idx.name for idx in idx_client.list_indexes()]:
                idx_client.delete_index(self.config.INDEX_NAME)
                print(f"[Indexer] Index '{self.config.INDEX_NAME}' supprimé")
        except Exception as e:
            print(f"[Indexer] Erreur suppression index: {e}")

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="search_result_title", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="search_result_link", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page_language", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="actionnaires", type=SearchFieldDataType.String),
            SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, retrievable=False,
                vector_search_dimensions=1536,
                vector_search_profile_name="default"
            )
        ]
        vect_name = "openai_vectorizer"
        vectorizer = AzureOpenAIVectorizer(
            vectorizer_name=vect_name,
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=self.config.AZURE_OPENAI_ENDPOINT,
                deployment_name=self.config.EMBEDDING_DEPLOYMENT_NAME,
                model_name=self.config.EMBEDDING_MODEL_NAME,
                api_key=self.config.AZURE_OPENAI_KEY
            )
        )
        vector_search = VectorSearch(
            vectorizers=[vectorizer],
            profiles=[VectorSearchProfile(
                name="default",
                algorithm_configuration_name="hnsw-config",
                vectorizer_name=vect_name
            )],
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")]
        )
        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="content"),
                        content_fields=[SemanticField(field_name="content")]
                    )
                )
            ],
            default_configuration_name="default"
        )
        idx = SearchIndex(
            name=self.config.INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            vectorizers=[vectorizer],
            semantic_search=semantic_search
        )
        idx_client.create_or_update_index(idx)
        print(f"[Indexer] Index '{self.config.INDEX_NAME}' (re)créé")

    def upload_to_index(self, docs=None):
        search_client = SearchClient(
            endpoint=self.config.AZURE_SEARCH_ENDPOINT,
            index_name=self.config.INDEX_NAME,
            credential=AzureKeyCredential(self.config.AZURE_SEARCH_KEY)
        )
        BATCH_SIZE = self.config.BATCH_UPLOAD
        if docs is None:
            # Charger selon le format
            if self.config.USE_JSONL:
                docs = []
                with open(self.config.JSONL_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        docs.append(json.loads(line))
            else:
                with open(self.config.JSON_PATH, "r", encoding="utf-8") as f:
                    docs = json.load(f)
        total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"[Indexer] Upload de {len(docs)} docs ({total_batches} batchs) dans Azure Search...")
        for b in tqdm(range(total_batches), desc="Upload batches"):
            batch = docs[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            self.exponential_backoff_retry(
                lambda: search_client.upload_documents(documents=batch),
                max_retries=6,
                base_delay=5
            )
        print("[Indexer] Upload terminé avec succès.")

    def normalize_doc(self, doc: dict) -> dict:
        """
        Convertit tous les Decimal en float (pour des besoins d'upload en streaming)
        parcours récursif simple pour listes et dicts.
        """
        def _fix(obj):
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: _fix(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_fix(v) for v in obj]
            return obj

        return _fix(doc)

    def upload_json_stream_to_index(self, json_path, search_client):
        BATCH_SIZE = self.config.BATCH_UPLOAD
        batch = []
        total = 0
        with open(json_path, "rb") as f:
            objects = ijson.items(f, "item")
            for doc in tqdm(objects, desc="Upload JSON streaming"):
                doc = self.normalize_doc(doc)
                batch.append(doc)
                if len(batch) == BATCH_SIZE:
                    self.exponential_backoff_retry(
                        lambda: search_client.upload_documents(documents=batch),
                        max_retries=6,
                        base_delay=5
                    )
                    total += len(batch)
                    batch = []
            if batch:
                self.exponential_backoff_retry(
                    lambda: search_client.upload_documents(documents=batch),
                    max_retries=6,
                    base_delay=5
                )
                total += len(batch)
        print(f"Upload terminé ({total} documents uploadés depuis {json_path})")

    def upload_jsonl_stream_to_index(self, jsonl_path, search_client):
        BATCH_SIZE = self.config.BATCH_UPLOAD
        batch = []
        total = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Upload JSONL streaming"):
                doc = self.normalize_doc(json.loads(line))
                batch.append(doc)
                if len(batch) == BATCH_SIZE:
                    self.exponential_backoff_retry(
                        lambda: search_client.upload_documents(documents=batch),
                        max_retries=6,
                        base_delay=5
                    )
                    total += len(batch)
                    batch = []
            if batch:
                self.exponential_backoff_retry(
                    lambda: search_client.upload_documents(documents=batch),
                    max_retries=6,
                    base_delay=5
                )
                total += len(batch)
        print(f"Upload terminé ({total} documents uploadés depuis {jsonl_path})")


    # LA PIPELINEEEEEEEE
    def pipeline(self, documents):
        chunked_docs = self.chunk_documents(documents)
        if self.config.EMBED_BEFORE_UPLOAD:
            chunked_docs = self.add_embeddings(chunked_docs)

        if self.config.SAVE_CHUNKS_LOCALLY:
            if self.config.USE_JSONL:
                self.save_chunks_to_jsonl(chunked_docs)
            else:
                self.save_chunks_to_json(chunked_docs)

        self.create_or_reset_index()
        # Upload selon config
        if self.config.SAVE_CHUNKS_LOCALLY:
            self.upload_to_index(chunked_docs if not self.config.USE_JSONL else None)
        else:
            self.upload_to_index(chunked_docs)
        print("\n[Pipeline] Terminé ! 🎉")

# --------------- MAIN SCRIPT ---------------

if __name__ == "__main__":
    config = Config()
    processor = DocumentProcessor(config)
    indexer = AzureIndexer(config)

    # Modes :
    if config.UPLOAD_ONLY:
        # Recréation index si voulu
        if config.RECREATE_INDEX:
            indexer.create_or_reset_index()

        search_client = SearchClient(
            endpoint=config.AZURE_SEARCH_ENDPOINT,
            index_name=config.INDEX_NAME,
            credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
        )
        # Streaming JSON array (gros .json)
        if config.UPLOAD_STREAMING and not config.USE_JSONL:
            indexer.upload_json_stream_to_index(config.JSON_PATH, search_client)
        # Streaming JSONL
        elif config.UPLOAD_STREAMING and config.USE_JSONL:
            indexer.upload_jsonl_stream_to_index(config.JSONL_PATH, search_client)
        # Lecture et upload classique (tout en RAM)
        else:
            indexer.upload_to_index()
        print("\n[Pipeline - Upload only] Terminé ! 🎉")
    else:
        # Pipeline complet
        documents = processor.read_documents()
        indexer.pipeline(documents)

