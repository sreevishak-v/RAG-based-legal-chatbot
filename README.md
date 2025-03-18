
# KELBot: Interactive Legal Chatbot⚖️

KELBot is an AI-powered  RAG Based chatbot designed to assist users in querying legal case data, specifically from the High Court of Kerala. Built with a combination of Streamlit for the user interface, FAISS for efficient vector-based retrieval, and a powerful language model (facebook/opt-350m) for natural language generation, LegalBot retrieves and presents detailed case information in a conversational manner. It supports queries about case outcomes, judge details, sections involved, and more, making it a valuable tool for legal research and exploration.

## Features
- Interactive Chat Interface: Built with Streamlit, allowing users to input queries and receive responses in a chat-like format.
- Accurate Case Retrieval: Utilizes FAISS for cosine similarity search, with exact case ID matching for precise lookups.
- Comprehensive Metadata: Retrieves and displays case details including case ID, court, date, judge, sections, outcome, and full text snippets.
- Human-Like Responses: Powered by facebook/opt-350m, a 350M-parameter model, for natural and context-aware replies.
- Expandable Details: Users can view raw retrieved case data with cosine similarity scores via an expander section.
- Flexible Query Handling: Supports specific queries (e.g., "Outcome of Case ID: CRL.MC NO. 284 OF 2024") and general questions (e.g., "What happened in 2016?").

Project Structure

```bash
LegalBot/
├── cleaned_data/          Directory containing preprocessed JSON case files
├── vector_store.faiss     FAISS index file for vector embeddings
├── metadata.json          Metadata file with case details
├── app.py                 Main Streamlit application script
├── requirements.txt       List of Python dependencies
└── README.txt             Project documentation
```

## Architecture
LegalBot follows a layered architecture for efficient query handling and response generation:


|       User Interface (Streamlit) |
|  - Chat Input                    |
```+----------------------------------+
|       User Interface (Streamlit) |
|  - Chat Input                    |
|  - Response Display              |
|  - Expander for Case Details     |
+----------------------------------+
            (User Query)
+----------------------------------+
|       Retrieval Layer            |
|  - FAISS Vector Search           |
|    - Exact Case ID Match         |
|    - Cosine Similarity (Top 3)   |
|  - Metadata Extraction           |
|    - Case ID, Judge, Full Text   |
+----------------------------------+
            (Retrieved Cases)
+----------------------------------+
|       Processing Layer           |
|  - facebook/opt-350m LLM         |
|    - Intent-Focused Prompt       |
|    - Natural Language Generation |
|  - Fallback Logic                |
+----------------------------------+
            (Generated Response)
+----------------------------------+
|       Output Layer               |
|  - Human-Like Response           |
|  - Detailed Case Data (Expander) |
+----------------------------------+
```

## Components
1. User Interface (Streamlit):
   - Provides an interactive chat interface for query input and response display.

2. Retrieval Layer:
   - FAISS: Indexes case embeddings for similarity search.
   - Exact Matching: Prioritizes exact case ID matches (e.g., "CRL.MC NO. 284 OF 2024").
   - Similarity: Ranks top 3 cases by cosine similarity if no exact match.

3. Processing Layer:
   - OPT-350m: Generates natural responses based on query intent (e.g., "outcome", "judge").
   - Prompt: Guides the LLM with context and intent detection.
   - Fallback: Rule-based responses if LLM output is inadequate.

4. Output Layer:
   - Delivers conversational replies and optional detailed case data.

## Dependencies
See requirements.txt for the full list:
- streamlit
- sentence-transformers
- faiss-cpu
- transformers
- torch
