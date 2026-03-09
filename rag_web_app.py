from flask import Flask, render_template, request, jsonify
from rag_procedure_app import RAGProcedureSuggestionApp
import os
import pandas as pd
from dotenv import load_dotenv
import gc
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

rag_app = None


def get_rag_app():
    global rag_app
    if rag_app is None:
        excel_path = "UPDATED PROCEDURES AND CONCERNS DATABASE .xlsx"
        api_key = os.getenv("OPENAI_API_KEY")
        rag_app = RAGProcedureSuggestionApp(
            excel_file_path=excel_path,
            api_key=api_key,
            use_local_model=True
        )
        gc.collect()
    return rag_app


@app.route("/")
def index():
    return render_template("rag_index.html")


@app.route("/health")
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "rag_app_initialized": rag_app is not None
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/refresh", methods=["POST"])
def refresh_embeddings():
    try:
        app_instance = get_rag_app()
        app_instance.refresh_embeddings()
        gc.collect()
        return jsonify({"message": "Embeddings refreshed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/suggest", methods=["POST"])
def semantic_search():
    try:
        user_query = request.json.get("query", "").strip()
        top_k = request.json.get("top_k", 5)

        if not user_query:
            return jsonify({"error": "Please provide a valid query"}), 400

        top_k = min(top_k, 20)

        app_instance = get_rag_app()

        raw_results = app_instance.semantic_search(user_query, top_k)

        if raw_results:
            best_score = raw_results[0][1]
        else:
            best_score = 0.0

        if best_score < 0.23:
            return jsonify({
                "semantic_results": [],
                "count": 0,
                "suggestions": "No relevant medical procedures found."
            })

        formatted = []
        for idx, score, text in raw_results:
            if idx >= len(app_instance.procedures_data):
                continue

            row = app_instance.procedures_data.iloc[idx]
            entry = {
                "similarity_score": round(score, 3),
                "sheet_source": row.get("Sheet_Source", "Unknown")
            }

            for col in app_instance.procedures_data.columns:
                if col == "Sheet_Source":
                    continue
                val = row.get(col)
                if pd.notna(val):
                    key = col.lower().replace(" ", "_").replace("-", "_")
                    entry[key] = str(val).strip()

            formatted.append(entry)

        if not formatted:
            return jsonify({
                "semantic_results": [],
                "count": 0,
                "suggestions": "No relevant procedures found."
            })

        suggestions_text = "\n".join([
            f"{i+1}. {proc.get('procedure', proc.get('procedure_name', 'Unknown Procedure'))} "
            f"(Score: {proc.get('similarity_score', 'N/A')})"
            for i, proc in enumerate(formatted)
        ])

        gc.collect()

        return jsonify({
            "semantic_results": formatted,
            "count": len(formatted),
            "suggestions": suggestions_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
