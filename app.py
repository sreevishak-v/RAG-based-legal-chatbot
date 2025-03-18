import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
import streamlit as st
from pathlib import Path
import re
from transformers import pipeline
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Folder paths
CLEANED_FOLDER = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/cleaned_data'
VECTOR_STORE_PATH = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/vector_store.faiss'
METADATA_PATH = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/metadata.json'

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load OPT-350m for better text generation
try:
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline('text-generation', model='facebook/opt-350m', device=device)
    logger.info("Loaded opt-350m on GPU" if device == 0 else "Loaded opt-350m on CPU")
except RuntimeError as e:
    logger.warning(f"GPU failed: {e}. Falling back to CPU.")
    generator = pipeline('text-generation', model='facebook/opt-350m', device=-1)

def normalize_sections(sections):
    cleaned = set()
    for sec in sections:
        sec = re.sub(r'read\s+with.*$', '', sec, flags=re.IGNORECASE).strip()
        if not re.match(r'^\d{4}/[A-Z]+/\d+$', sec) and not re.match(r'^\d{4}$', sec):
            cleaned.add(sec.upper())
    return list(cleaned)

def load_vector_store():
    try:
        index = faiss.read_index(VECTOR_STORE_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        for meta in metadata:
            if not meta.get('case_id') or meta['case_id'].startswith("Unknown"):
                case_id_match = re.search(r'(Crl\.MC\.No\.|CRL\.MC\s+NO\.|CC|SC|W\.P\.)\s*[\d/]+\s*(?:of\s*\d+)?\s*(?:\([^)]*\))?', meta['full_text'], re.IGNORECASE)
                meta['case_id'] = case_id_match.group(0).strip() if case_id_match else f"Unknown_{meta['file'][:10]}"
            # Ensure judge field exists
            if 'judge' not in meta:
                judge_match = re.search(r'(?:Justice|Honourable\s+Mr\.?|Mrs\.?)\s+[A-Za-z\s]+', meta['full_text'], re.IGNORECASE)
                meta['judge'] = judge_match.group(0) if judge_match else "Not specified"
        return index, metadata
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None

def query_vector_store(query, index, metadata, top_k=20):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [
        {
            'metadata': metadata[idx],
            'distance': float(dist),
            'cosine_similarity': 1 - (dist / 2)
        }
        for idx, dist in zip(indices[0], distances[0])
    ]
    # Exact case ID match
    case_id_match = re.search(r'(Crl\.MC\.No\.|CRL\.MC\s+NO\.)\s*\d+\s*(?:of|OF)\s*\d+', query, re.IGNORECASE)
    if case_id_match:
        case_id = case_id_match.group(0).upper().replace(" ", "")
        exact_match = [r for r in results if r['metadata']['case_id'].upper().replace(" ", "") == case_id]
        if exact_match:
            logger.info(f"Exact match found for {case_id}")
            return exact_match[:1]
        logger.warning(f"No exact match for {case_id} in top {top_k} results")
    # Fallback to top 3 by similarity
    return sorted(results, key=lambda x: x['cosine_similarity'], reverse=True)[:3]

def generate_natural_response(query, top_results):
    if not top_results:
        return f"Sorry, I couldn’t find any details for '{query}' in the database. Please check the case ID or try a different question."

    # Use all metadata fields in context
    context = "\n".join([
        f"Case ID: {r['metadata']['case_id']} | Court: {r['metadata']['court']} | Date: {r['metadata']['date']} | Judge: {r['metadata']['judge']} | Sections: {', '.join(r['metadata']['sections'])} | Outcome: {r['metadata']['outcome']} | Full Text Snippet: {r['metadata']['full_text'][:200]}..."
        for r in top_results
    ])

    # Refined prompt
    prompt = (
        f"System: You are a legal assistant providing detailed, human-like answers about court cases. "
        f"Focus on the user's query intent (e.g., 'outcome' for case results, 'judge' for judge details). "
        f"Use all available context—case ID, court, date, judge, sections, outcome, and full text—to craft a natural, accurate response. "
        f"If an exact case ID is asked, prioritize that case’s details. Avoid technical jargon and invented info.\n"
        f"User Query: {query}\n"
        f"Context:\n{context}\n"
        "Response:"
    )
    
    try:
        response = generator(prompt, max_new_tokens=150, num_return_sequences=1, do_sample=True, temperature=0.7)[0]['generated_text']
        response = response[len(prompt):].strip()
        response = re.sub(r'\n.*$', '', response)
        
        # Fallback if response is inadequate
        if len(response) < 20 or not any(r['metadata']['case_id'] in response for r in top_results):
            case = top_results[0]
            if "outcome of case id" in query.lower():
                return f"The outcome of {case['metadata']['case_id']} was that it was {case['metadata']['outcome']} on {case['metadata']['date']} at {case['metadata']['court']}, presided over by {case['metadata']['judge']}."
            elif "judge" in query.lower():
                return f"The judge for {case['metadata']['case_id']} was {case['metadata']['judge']} in a case decided on {case['metadata']['date']} at {case['metadata']['court']}."
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                year = year_match.group(1)
                relevant = [r for r in top_results if year in r['metadata']['date']]
                if relevant:
                    outcomes = [f"{r['metadata']['case_id']} was {r['metadata']['outcome']} by {r['metadata']['judge']}" for r in relevant]
                    return f"In {year}, at the High Court of Kerala, I found: {', '.join(outcomes)}."
                return f"I couldn’t find cases from {year} matching your query."
            return f"For {case['metadata']['case_id']}, the outcome was {case['metadata']['outcome']} on {case['metadata']['date']} at {case['metadata']['court']} with {case['metadata']['judge']} presiding."
        return response
    except Exception as e:
        case = top_results[0]
        return f"I had trouble generating a response ({e}). For {case['metadata']['case_id']}, it was {case['metadata']['outcome']} on {case['metadata']['date']} at {case['metadata']['court']} with {case['metadata']['judge']} presiding."

def main():
    index, metadata = load_vector_store()
    if index is None or metadata is None:
        return

    st.title("KELBot: Interactive Legal Chatbot⚖️")
    st.write("Ask anything about Kerala legal cases👨‍⚖️")

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Enter your query:")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Processing..."):
            top_results = query_vector_store(query, index, metadata)
            response = generate_natural_response(query, top_results)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        # Show all metadata in expander
        if top_results:
            with st.expander("View Retrieved Case Details"):
                st.write("**Most Relevant Cases:**")
                for r in top_results:
                    st.write(
                        f"- **Case ID:** {r['metadata']['case_id']}  \n"
                        f"  **Court:** {r['metadata']['court']}  \n"
                        f"  **Date:** {r['metadata']['date']}  \n"
                        f"  **Judge:** {r['metadata']['judge']}  \n"
                        f"  **Sections:** {', '.join(r['metadata']['sections'])}  \n"
                        f"  **Outcome:** {r['metadata']['outcome']}  \n"
                        f"  **Full Text Snippet:** {r['metadata']['full_text'][:200]}...  \n"
                        f"  **Cosine Similarity:** {r['cosine_similarity']:.3f}"
                    )

if __name__ == "__main__":
    main()