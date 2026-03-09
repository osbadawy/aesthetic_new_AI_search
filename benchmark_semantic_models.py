#!/usr/bin/env python3
"""
Benchmark Semantic Models for Aesthetic RAG Pipeline
---------------------------------------------------
This script benchmarks multiple semantic search models (CPU-friendly)
using your existing dataset:
  'UPDATED PROCEDURES AND CONCERNS DATABASE.xlsx'
Benchmarks semantic search models using the new DB:
  UPDATED PROCEDURES AND CONCERNS DATABASE.xlsx

It measures:
 - Model loading time
 - Embedding generation time
 - Query response time
 - Embedding accuracy proxy (cosine similarity check)

Outputs:
 - benchmark_results.csv  → performance summary
 - *_embeddings.npy       → embeddings for each model
 - procedure_embeddings.pkl (optional) for RAG integration

Usage:
    (venv) python benchmark_semantic_models.py
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# Force CPU usage to avoid MPS compatibility issues on Apple Silicon
if torch.backends.mps.is_available():
    print("🍎 Apple Silicon detected - forcing CPU usage to avoid MPS compatibility issues")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    torch.set_default_device('cpu')


print("Starting...")
# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
EXCEL_FILE = "UPDATED PROCEDURES AND CONCERNS DATABASE.xlsx"
SAVE_PKL = True  # Set False to skip overwriting procedure_embeddings.pkl

MODELS = {
    "sentence-transformers/static-similarity-mrl-multilingual-v1": None
}


# ---------------------------------------------------------------------
# LOAD PROCEDURES FROM EXCEL (Updated to match web app approach)
# ---------------------------------------------------------------------
print(f"Loading procedures from {EXCEL_FILE} ...")
try:
    df = pd.read_excel(EXCEL_FILE, sheet_name=None)
except Exception as e:
    raise SystemExit(f"ERROR: Could not read Excel file '{EXCEL_FILE}': {e}")

def create_procedure_text(row, sheet_name):
    """
    Create searchable text from procedure row - matches web app approach
    """
    text_parts = []
    
    for column, value in row.items():
        if pd.notna(value):
            # Check if column contains procedure-related content
            col_lower = column.lower()
            if any(keyword in col_lower for keyword in ['procedure', 'concern', 'verbatim', 'treatment', 'description']):
                text_parts.append(str(value).strip())
            else:
                # For other columns, include with label
                text_parts.append(f"{column}: {str(value).strip()}")
    
    return " | ".join(text_parts)

# Extract complete procedure records (matching web app approach)
texts = []
procedures_data = []

for sheet_name, sheet in df.items():
    print(f"  Processing sheet: {sheet_name} ({len(sheet)} rows)")
    
    for idx, row in sheet.iterrows():
        # Create procedure record
        procedure_record = row.to_dict()
        procedure_record['Sheet_Source'] = sheet_name
        procedures_data.append(procedure_record)
        
        # Create combined text for embedding
        combined_text = create_procedure_text(row, sheet_name)
        texts.append(combined_text)

if not texts:
    raise SystemExit("No procedure texts found in the Excel file.")

print(f"✅ Loaded {len(texts)} complete procedure records from {len(df)} sheets")

# Limit for benchmarking to keep runtime reasonable
MAX_TEXTS = 370
if len(texts) > MAX_TEXTS:
    # texts = texts[:MAX_TEXTS]
    # procedures_data = procedures_data[:MAX_TEXTS]
    print(f"⚠️  Limited to {MAX_TEXTS} procedures for benchmarking")

print(f"Using {len(texts)} procedure records for benchmarking")

# ---------------------------------------------------------------------
# FUNCTION: Benchmark model
# ---------------------------------------------------------------------
def benchmark_model(model_name):
    print(f"\n🚀 Benchmarking model: {model_name}")
    results = {}

    # ---- Load model ----
    try:
        start_time = time.time()
        # Force CPU device to avoid MPS compatibility issues
        model = SentenceTransformer(model_name, device='cpu')
        load_time = time.time() - start_time
        results["load_time_sec"] = round(load_time, 2)
        print(f"  Loaded model in {results['load_time_sec']}s (using CPU)")
    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        results["load_time_sec"] = "ERROR"
        results["embedding_time_sec"] = "ERROR"
        results["query_time_sec"] = "ERROR"
        results["max_similarity_score"] = "ERROR"
        results["model_size_mb"] = "N/A"
        return results

    # ---- Generate embeddings ----
    try:
        start_time = time.time()
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embed_time = time.time() - start_time
        results["embedding_time_sec"] = round(embed_time, 2)
        print(f"  Generated embeddings in {results['embedding_time_sec']}s")
    except Exception as e:
        print(f"  ERROR generating embeddings with {model_name}: {e}")
        results["embedding_time_sec"] = "ERROR"
        results["query_time_sec"] = "ERROR"
        results["max_similarity_score"] = "ERROR"
        results["model_size_mb"] = "N/A"
        return results

    # ---- Query response time ----
    try:
        # Test with multiple realistic queries
        test_queries = [
            "I want to lift my cheeks and reduce sagging around the jawline",
            "How can I reduce wrinkles around my eyes",
            "I'm looking for non-surgical facial rejuvenation options",
            "What procedures help with skin tightening"
        ]
        
        max_similarity = 0
        total_query_time = 0
        
        for query in test_queries:
            start_time = time.time()
            query_embedding = model.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(query_embedding, embeddings)
            query_time = time.time() - start_time
            total_query_time += query_time
            max_similarity = max(max_similarity, float(np.max(similarities)))
        
        results["query_time_sec"] = round(total_query_time / len(test_queries), 4)
        results["max_similarity_score"] = round(max_similarity, 4)
        print(f"  Query responded in {results['query_time_sec']}s avg (max sim = {results['max_similarity_score']})")
    except Exception as e:
        print(f"  ERROR during query timing for {model_name}: {e}")
        results["query_time_sec"] = "ERROR"
        results["max_similarity_score"] = "ERROR"

    # ---- Save embeddings ----
    safe_name = model_name.replace("/", "_").replace(":", "_")
    emb_file = f"{safe_name}_embeddings.npy"
    try:
        np.save(emb_file, embeddings)
        print(f"  💾 Saved embeddings to {emb_file}")
    except Exception as e:
        print(f"  ERROR saving embeddings for {model_name}: {e}")

    # ---- Optionally update procedure_embeddings.pkl for RAG (Updated format) ----
    if SAVE_PKL:
        try:
            # Convert embeddings to list format for compatibility
            embeddings_list = embeddings.tolist()

            
            # Create embeddings data structure matching web app expectations
            embeddings_data = {
                'embeddings': embeddings_list,
                'texts': texts,
                'procedures_data': procedures_data,  # Include procedure records
                'model': model_name,
                'data_hash': hash(str(pd.concat(df.values()).values.tobytes()))
            }
            
            with open("procedure_embeddings.pkl", "wb") as f:
                pickle.dump(embeddings_data, f)
            print("  ✅ procedure_embeddings.pkl updated for RAG pipeline (new format)")
        except Exception as e:
            print(f"  ERROR saving procedure_embeddings.pkl: {e}")

    # ---- Estimate model size (approx) ----
    model_size_mb = "N/A"
    try:
        # Try to locate model cache folder for sentence-transformers / huggingface
        possible_cache_dirs = [
            os.path.expanduser("~/.cache/torch/sentence_transformers"),
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/.cache/huggingface/transformers"),
        ]
        total_bytes = 0
        for cache_dir in possible_cache_dirs:
            if os.path.isdir(cache_dir):
                for root, _, files in os.walk(cache_dir):
                    for file in files:
                        # crude filter: include files that mention the model name (last segment) if present
                        if model_name.split("/")[-1] in root or model_name.split("/")[-1] in file:
                            try:
                                total_bytes += os.path.getsize(os.path.join(root, file))
                            except Exception:
                                pass
        if total_bytes > 0:
            model_size_mb = round(total_bytes / (1024 * 1024), 1)
    except Exception:
        model_size_mb = "N/A"

    results["model_size_mb"] = model_size_mb

    return results

# ---------------------------------------------------------------------
# RUN BENCHMARKS
# ---------------------------------------------------------------------
records = []
for model_key in MODELS.keys():
    res = benchmark_model(model_key)
    res["model"] = model_key
    records.append(res)

# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------
cols = ["model", "model_size_mb", "load_time_sec", "embedding_time_sec", "query_time_sec", "max_similarity_score"]
results_df = pd.DataFrame(records)
results_df = results_df[[c for c in cols if c in results_df.columns]]
results_df.to_csv("benchmark_results.csv", index=False)

print("\n📊 Benchmark complete! Results saved to benchmark_results.csv")
print(results_df.to_string(index=False))
