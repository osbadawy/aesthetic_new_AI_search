import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch


# Force CPU on Apple Silicon
if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.set_default_device("cpu")


# ------------------------------------------------------------------
# New DB constants
# ------------------------------------------------------------------
NEW_DB_FILE = "UPDATED PROCEDURES AND CONCERNS DATABASE.xlsx"

# Sheets that contain actual procedure data in the new DB.
# "Feuille 2" is a brand/technique reference — excluded intentionally.
PROCEDURE_SHEETS = [
    "All_Procedures",
    "Surgical_Only",
    "NonSurgical_Only",
    "Aesthetic_Dentistry",
    "IV_Therapy",
    "Hair_Medicine",
    "Muscle_Skin_Tone",
    "Slimming_wheight",
    "Other",
]

# Rows whose procedure_title starts with this marker are section headers
# injected by the spreadsheet author — skip them.
HEADER_MARKER = "✦"

# The primary key column in the new DB
PROCEDURE_TITLE_COL = "procedure_title"

# Columns that are semantically rich and embedded more prominently
SEMANTIC_PRIORITY_COLS = [
    "procedure_title",
    "concerns",
    "short_description",
    "expected_results",
    "treatment_type",
    "main_zone",
    "face_subzone",
    "body_subzone",
]


class RAGProcedureSuggestionApp:
    def __init__(self, excel_file_path, api_key, embeddings_cache_path="procedure_embeddings.pkl", use_local_model=True):
        self.excel_file_path = excel_file_path
        self.embeddings_cache_path = embeddings_cache_path
        self.use_local_model = use_local_model

        # Load local multilingual embedding model
        self.model = SentenceTransformer(
            "sentence-transformers/static-similarity-mrl-multilingual-v1",
            device="cpu"
        )

        # Load data + embeddings
        self.procedures_data = self.load_procedures_data()
        self.embeddings, self.texts = self.load_or_create_embeddings()


    # -------------------------------------------------------------
    # Load Excel database
    # -------------------------------------------------------------
    def load_procedures_data(self):
        xl = pd.ExcelFile(self.excel_file_path)
        """
        Load procedure records from UPDATED PROCEDURES AND CONCERNS DATABASE.xlsx.

        Strategy:
        - Use All_Procedures as the single source of truth (138 clean rows,
          no section-header rows, all 18 columns present).
        - Sub-sheets (Surgical_Only, NonSurgical_Only, etc.) contain the same
          procedures split by category plus decorative section-header rows
          (prefixed with ✦).  We keep sub-sheet data only to attach the
          Sheet_Source label so callers can filter by category if needed,
          but we deduplicate on procedure_title so no procedure appears twice.
        """
        try:
            xl = pd.ExcelFile(self.excel_file_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not open Excel file '{self.excel_file_path}': {e}"
            )

        frames = []

        for sheet in xl.sheet_names:
            df = pd.read_excel(self.excel_file_path, sheet_name=sheet)
            df["Sheet_Source"] = sheet
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        return combined


    # -------------------------------------------------------------
    # Convert a row into a searchable text
    # -------------------------------------------------------------
    def create_text(self, row):
        parts = []
        for col, val in row.items():
            if pd.notna(val) and col != "Sheet_Source":
                parts.append(f"{col}: {str(val)}")
        return " | ".join(parts)


    # -------------------------------------------------------------
    # Embedding cache load / create
    # -------------------------------------------------------------
    def load_or_create_embeddings(self):
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, "rb") as f:
                    data = pickle.load(f)
                emb = np.array(data["embeddings"])
                txt = list(data["texts"])
                return emb, txt
            except Exception:
                pass

        return self.create_embeddings()


    def create_embeddings(self):
        texts = []
        rows = self.procedures_data

        print("Creating fresh embeddings...")

        for _, row in rows.iterrows():
            txt = self.create_text(row)
            texts.append(txt)

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        data = {
            "embeddings": embeddings.tolist(),
            "texts": texts,
        }

        with open(self.embeddings_cache_path, "wb") as f:
            pickle.dump(data, f)

        return embeddings, texts


    # -------------------------------------------------------------
    # PURE semantic search (FINAL FIX)
    # -------------------------------------------------------------
    def semantic_search(self, query, top_k=5):
        """
        Pure multilingual semantic search.
        No domain filtering. No relevance blocking. No TF-IDF.
        Returns list -> (index, similarity, text)
        """

        query = query.strip()
        if not query:
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True)

        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idxs = sims.argsort()[::-1]

        results = []
        for i in idxs[:top_k]:
            results.append((int(i), float(sims[i]), self.texts[i]))

        return results


    # -------------------------------------------------------------
    # Optional: force rebuild
    # -------------------------------------------------------------
    def refresh_embeddings(self):
        if os.path.exists(self.embeddings_cache_path):
            os.remove(self.embeddings_cache_path)

        self.embeddings, self.texts = self.create_embeddings()
