import fitz 
import gradio as gr
from transformers import pipeline
from typing import List
import tempfile
import re
from datetime import datetime
import os

# Chargement du pipeline QA francophone
qa = pipeline(
    "question-answering",
    model="etalab-ia/camembert-base-squadFR-fquad-piaf",
    tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf",
    use_fast=False
)

# Extraction du texte par page
def extraire_chunks(pdf_file) -> List[str]:
    try:
        with open(pdf_file.name, "rb") as f:
            pdf_bytes = f.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return [page.get_text() for page in doc if page.get_text().strip()]
    except Exception as e:
        return [f"Erreur lors de l'extraction du PDF : {str(e)}"]

# Nettoyage du nom de fichier
def nettoyer_nom_fichier(question: str) -> str:
    question_simplifiee = re.sub(r'[^a-zA-Z0-9_\- ]', '', question)[:50].strip().replace(" ", "_")
    if not question_simplifiee:
        question_simplifiee = "reponse"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{question_simplifiee}_{timestamp}.txt"

# Réponse à partir des chunks
def repondre_pdf(question, pdf_file):
    if not pdf_file:
        return "❌ Aucun fichier PDF fourni.", "", None
    
    chunks = extraire_chunks(pdf_file)
    
    if not chunks or "Erreur" in chunks[0]:
        return (chunks[0] if chunks else "❌ Impossible de lire le contenu du PDF."), "", None

    reponses = []
    for chunk in chunks:
        try:
            result = qa(question=question, context=chunk)
            result["context"] = chunk
            reponses.append(result)
        except Exception:
            continue

    if not reponses:
        return "❌ Aucune réponse trouvée.", "", None

    # Trier par score décroissant et prendre les 3 meilleurs
    reponses.sort(key=lambda x: x['score'], reverse=True)
    top_reponses = reponses[:3]

    resume = "✅ **Top réponses :**\n\n"
    extraits_complets = ""

    for idx, rep in enumerate(top_reponses, start=1):
        extrait_court = rep["context"].replace("\n", " ").strip()
        if len(extrait_court) > 500:
            extrait_court = extrait_court[:500] + "..."
        
        resume += (
            f"🔹 **Réponse {idx}** : {rep['answer']}\n"
            f"📊 Confiance : {rep['score']:.2f}\n"
            f"📄 Aperçu : {extrait_court}\n\n"
        )

        extraits_complets += (
            f"🔹 Réponse {idx} (score : {rep['score']:.2f}) : {rep['answer']}\n"
            f"📜 Extrait complet :\n{rep['context'].strip()}\n\n{'-'*60}\n\n"
        )

    # Créer un fichier exportable avec nom propre
    nom_fichier = nettoyer_nom_fichier(question)
    chemin_temp = os.path.join(tempfile.gettempdir(), nom_fichier)
    with open(chemin_temp, "w", encoding="utf-8") as f:
        f.write(extraits_complets)

    return resume.strip(), extraits_complets.strip(), chemin_temp

# Interface Gradio
demo = gr.Interface(
    fn=repondre_pdf,
    inputs=[
        gr.Textbox(label="❓ Question en français"),
        gr.File(label="📄 Fichier PDF", file_types=[".pdf"])
    ],
    outputs=[
        gr.Textbox(label="📌 Résumé des réponses", lines=10),
        gr.Textbox(label="📜 Extraits complets", lines=20, max_lines=30, show_copy_button=True),
        gr.File(label="⬇️ Télécharger les extraits complets")
    ],
    title="Agent de lecture PDF (FR)",
    description="Pose une question sur un PDF en français. Affiche les meilleures réponses et permet d’exporter les extraits."
)

if __name__ == "__main__":
    demo.launch()
