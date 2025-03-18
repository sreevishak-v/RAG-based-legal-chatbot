import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
from pathlib import Path
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Folder paths
CLEANED_FOLDER = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/cleaned_data'
VECTOR_STORE_PATH = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/vector_store.faiss'
METADATA_PATH = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/metadata.json'

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_sections(sections):
    cleaned = set()
    for sec in sections:
        sec = re.sub(r'read\s+with.*$', '', sec, flags=re.IGNORECASE).strip()
        if not re.match(r'^\d{4}/[A-Z]+/\d+$', sec) and not re.match(r'^\d{4}$', sec):  # Exclude citations and years
            cleaned.add(sec.upper())
    return list(cleaned)

def create_vector_store():
    json_files = [f for f in os.listdir(CLEANED_FOLDER) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} cleaned JSON files.")

    documents = []
    metadata = []
    for json_file in json_files:
        file_path = os.path.join(CLEANED_FOLDER, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            full_text = data['full_text']

            # Fix case_id
            if not data['case_id'] or data['case_id'] in ["Unknown", "CC 2015/"]:
                case_id_match = re.search(r'(Crl\.MC\.No\.|CRL\.MC\s+NO\.|CC|SC|W\.P\.)\s*[\d/]+\s*(?:of\s*\d+)?\s*(?:\([^)]*\))?', full_text, re.IGNORECASE)
                data['case_id'] = case_id_match.group(0).strip() if case_id_match else f"Unknown_{json_file[:10]}"

            # Fix date
            date_match = re.search(r'Dated\s+this\s+the\s+(\d{1,2}(?:ST|ND|RD|TH)?\s+DAY\s+OF\s+[A-Z]+\s*,\s*\d{4})', full_text, re.IGNORECASE)
            data['date'] = date_match.group(1).strip().upper() if date_match else data['date'].upper()

            # Fix sections
            data['sections'] = normalize_sections(data['sections'])

            # Fix outcome
            outcome_match = re.search(r'(quashed|granted|dismissed|allowed|disposed|rejected|bail)', full_text, re.IGNORECASE)
            data['outcome'] = outcome_match.group(0).lower() if outcome_match else data['outcome'][:50].lower()

            # Embedding text
            text = (
                f"{data['date']} {data['date']} "
                f"{data['court']} "
                f"{' '.join(data['sections'])} {' '.join(data['sections'])} "
                f"{data['outcome']} {data['outcome']} "
                f"{data['case_id']}"
            )
            documents.append(text)
            metadata.append({
                'file': json_file,
                'case_id': data['case_id'],
                'court': data['court'],
                'date': data['date'],
                'judge': data['judge'],
                'sections': data['sections'],
                'outcome': data['outcome'],
                'full_text': full_text[:500]
            })
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    logger.info("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, VECTOR_STORE_PATH)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    logger.info(f"Vector store saved to {VECTOR_STORE_PATH}, metadata to {METADATA_PATH}")

def query_vector_store(query, top_k=20):  # Further increased top_k
    index = faiss.read_index(VECTOR_STORE_PATH)
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [{'metadata': metadata[idx], 'distance': float(dist)} for idx, dist in zip(indices[0], distances[0])]
    return results

def filter_results(query, results):
    filtered = []
    query_lower = query.lower()

    if "2015" in query_lower and "quashed" in query_lower:
        filtered = [r for r in results if "2015" in r['metadata']['date'].lower() and "quash" in r['metadata']['outcome'].lower()]
        logger.debug(f"2015 quashed cases found: {len(filtered)}")
    elif "section 498a" in query_lower:
        filtered = [r for r in results if "498A" in r['metadata']['sections']]
        logger.debug(f"498A cases found: {len(filtered)}")
    elif "2024" in query_lower and "high court of kerala" in query_lower:
        filtered = [r for r in results if "2024" in r['metadata']['date'].lower() and "high court of kerala" in r['metadata']['court'].lower()]
        logger.debug(f"2024 High Court cases found: {len(filtered)}")
    
    return filtered[:5] if filtered else results[:3]

def rag_chatbot(query):
    results = query_vector_store(query)
    filtered_results = filter_results(query, results)
    
    if not filtered_results:
        return f"No relevant cases found for '{query}'."

    context = []
    for r in filtered_results:
        case_info = (
            f"Case ID: {r['metadata']['case_id']} "
            f"| Court: {r['metadata']['court']} "
            f"| Date: {r['metadata']['date']} "
            f"| Sections: {', '.join(r['metadata']['sections'])} "
            f"| Outcome: {r['metadata']['outcome']}"
        )
        context.append(case_info)
    
    response = f"Based on '{query}':\n" + "\n".join(context)
    return response

if __name__ == "__main__":
    create_vector_store()

    sample_queries = [
        "What cases were quashed in 2015?",
        "Find cases under Section 498A",
        "Outcomes for High Court of Kerala in 2024"
    ]
    
    for query in sample_queries:
        logger.info(f"\nQuery: {query}")
        response = rag_chatbot(query)
        logger.info(f"Response: {response}")