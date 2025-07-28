
# app.py
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import os
import json
import time
from utils import process_documents

app = Flask(__name__)
app.secret_key = "adobe_secret"
UPLOAD_FOLDER = "uploads"
OUTPUT_FILE = "output/result.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("output", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        persona = request.form.get("persona")
        job = request.form.get("job")
        files = request.files.getlist("documents")

        if not persona or not job or len(files) < 3:
            flash("Please enter all fields and upload at least 3 PDFs.")
            return redirect(url_for("index"))

        pdf_paths = []
        for f in files:
            filepath = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(filepath)
            pdf_paths.append(filepath)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        result = process_documents(pdf_paths, persona, job, timestamp)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2, ensure_ascii=False)

        return render_template("index.html", result=result, persona=persona, job=job)

    return render_template("index.html")

@app.route("/download")
def download():
    return send_file(OUTPUT_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)


# ============================
# utils.py
import fitz
from sentence_transformers import SentenceTransformer, util
import os

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_sections_from_pdf(path):
    doc = fitz.open(path)
    sections = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if len(text) >= 30:
                sections.append({
                    "doc": os.path.basename(path),
                    "page": i + 1,
                    "text": text
                })
    return sections


def score_relevance(texts, query):
    text_embeddings = model.encode([t["text"] for t in texts], convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, text_embeddings)[0]
    for i, s in enumerate(scores):
        texts[i]["score"] = float(s)
    return sorted(texts, key=lambda x: x["score"], reverse=True)


def process_documents(pdf_paths, persona, job, timestamp):
    all_sections = []
    for path in pdf_paths:
        sections = get_sections_from_pdf(path)
        all_sections.extend(sections)

    full_query = f"{persona} needs to {job}"
    ranked_sections = score_relevance(all_sections, full_query)[:10]

    result = {
        "metadata": {
            "documents": [os.path.basename(p) for p in pdf_paths],
            "persona": persona,
            "job": job,
            "timestamp": timestamp
        },
        "extracted_sections": [
            {
                "document": s["doc"],
                "page": s["page"],
                "section_title": s["text"].split("\n")[0][:50],
                "importance_rank": i + 1,
                "score": round(s["score"], 4)
            } for i, s in enumerate(ranked_sections)
        ],
        "subsection_analysis": [
            {
                "document": s["doc"],
                "page": s["page"],
                "refined_text": s["text"][:300]
            } for s in ranked_sections
        ]
    }

    return result


# ============================
# templates/index.html
<!DOCTYPE html>
<html>
<head>
    <title>Adobe Round 1B</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="container py-4">
    <h2>Persona-Driven PDF Extractor</h2>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="alert alert-danger">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <form method="POST" enctype="multipart/form-data">
        <div class="mb-3">
            <label class="form-label">Persona</label>
            <input type="text" name="persona" class="form-control" required value="{{ persona or '' }}">
        </div>
        <div class="mb-3">
            <label class="form-label">Job to be done</label>
            <input type="text" name="job" class="form-control" required value="{{ job or '' }}">
        </div>
        <div class="mb-3">
            <label class="form-label">Upload PDFs (min 3)</label>
            <input type="file" name="documents" class="form-control" multiple accept="application/pdf" required>
        </div>
        <button type="submit" class="btn btn-primary">Process</button>
    </form>

    {% if result %}
    <hr>
    <h4>Extracted Sections</h4>
    <ul class="list-group">
        {% for s in result.extracted_sections %}
        <li class="list-group-item">
            <b>{{ s.section_title }}</b> ({{ s.document }} - Page {{ s.page }}) - Rank {{ s.importance_rank }}, Score: {{ s.score }}
        </li>
        {% endfor %}
    </ul>
    <hr>
    <h4>Subsection Analysis</h4>
    <ul class="list-group">
        {% for s in result.subsection_analysis %}
        <li class="list-group-item">
            <b>{{ s.document }} - Page {{ s.page }}</b><br>
            {{ s.refined_text }}...
        </li>
        {% endfor %}
    </ul>
    <br>
    <a class="btn btn-success" href="/download">Download JSON</a>
    {% endif %}
</body>
</html>


# ============================
# requirements.txt
Flask==2.3.2
sentence-transformers==2.2.2
PyMuPDF==1.22.3
torch>=1.10


# ============================
# Dockerfile
FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
