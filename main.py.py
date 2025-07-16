import os
import glob
import json
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
    Centralise tous les paramètres du script.
    - CHUNK_TOK : nombre de tokens max par chunk (dépend du modèle d'embedding).
      Par défaut 512, possible jusqu'à 8191 tokens pour text-embedding-ada-002 (OpenAI/Azure).
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
        self.CHUNK_TOK = int(os.getenv("CHUNK_TOK", "512"))  # <---- Override possible, cf. doc
        self.CHUNK_OVER = 0
        self.MIN_CHUNK_TOK = 200
        self.EMBEDDING_MODEL = "text-embedding-ada-002"
        self.EMBED_BATCH = 430
        self.BATCH_UPLOAD = 975
        self.JSON_PATH = "chunked_docs_final.json"
        self.ERROR_LOG_PATH = "failed_uploads.log"
        self.EMBED_BEFORE_UPLOAD = os.getenv("EMBED_BEFORE_UPLOAD", "False").lower() == "true"
        self.SUPPORTED_EXTENSIONS = [".csv", ".txt"]  # Peut être étendu à ".pdf"
        self.CONTENT_FIELDS = [
            "search_result_title", "search_result_snippet", "page_description", "page_content"
        ]
        self.METADATA_FIELDS = [
            "search_result_link", "page_language"
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
                continue  # skip actionnaire.csv ici, logique spécifique
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
                resp = oaiclient.embeddings.create(input=contents, model=self.config.EMBEDDING_MODEL)
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

    def create_or_reset_index(self):
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
            SimpleField(name="search_result_link", type=SearchFieldDataType.String, filterable=True, retrievable=True),
            SimpleField(name="page_language", type=SearchFieldDataType.String, filterable=True, retrievable=True),
            SimpleField(name="actionnaires", type=SearchFieldDataType.String, retrievable=True),
            SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True, retrievable=True),
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
                deployment_name=self.config.EMBEDDING_MODEL,
                model_name=self.config.EMBEDDING_MODEL,
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

    def upload_to_index(self):
        search_client = SearchClient(
            endpoint=self.config.AZURE_SEARCH_ENDPOINT,
            index_name=self.config.INDEX_NAME,
            credential=AzureKeyCredential(self.config.AZURE_SEARCH_KEY)
        )
        BATCH_SIZE = self.config.BATCH_UPLOAD
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

    def pipeline(self, documents):
        chunked_docs = self.chunk_documents(documents)
        if self.config.EMBED_BEFORE_UPLOAD:
            chunked_docs = self.add_embeddings(chunked_docs)
        self.save_chunks_to_json(chunked_docs)
        self.create_or_reset_index()
        self.upload_to_index()
        print("\n[Pipeline] Terminé ! 🎉")

# --------------- MAIN SCRIPT ---------------

if __name__ == "__main__":
    config = Config()
    processor = DocumentProcessor(config)
    indexer = AzureIndexer(config)
    documents = processor.read_documents()
    indexer.pipeline(documents)
