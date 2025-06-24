🌟 TerShine
A Python-based toolkit for scraping, vector-indexing, and querying product and guide data. Ideal for building question‑answering tools or recommendation systems utilizing both structured and unstructured data.

🚀 Features
Web scraping of product listings, washing guides, and transcripts.

Vector indexing for retrieving semantically relevant information.

Multiple backends supported: basic, LangGraph, and custom logic.

Validation and testing functionality for development.

Front‑end component for interactive querying.

📂 Repository Structure
bash
Copy
Edit
tershine/
├── backend.py                # Main backend logic
├── backend_langgraph.py      # Backend alternative experimentation using LangGraph
├── aiscrapesubquestion.py    # Testing Sub Question Decomposition from Big Question
├── extract_wg_links.py       # Extract washing guide URLs from Tershine webpage
├── get_products.py           # Scrape raw product data
├── validation.py             # For testing purposes, can ignore.
├── test.py                   # [Incomplete] Small experiment on fine tuning AI model on voice data from youtube video. 
├── requirements.txt         # Python dependencies
├── vector_index/            # Vector index data and helper code
├── frontend/                # Web UI and assets for interaction
├── *.txt/.json/.pyc         # Raw data dumps, transcripts, caches
└── .gitignore, .gitattributes
🛠 Setup & Installation
Clone the repo:

bash
Copy
Edit
git clone https://github.com/briannoelkesuma/tershine.git
cd tershine
Create a virtual environment (recommended):

bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🔧 Usage
1. Scrape Products & Guides
get_products.py: identify and save product links/data.

extract_wg_links.py: extract washing-guide URLs from sources.

2. Scrape & Process Datasets
aiscrapesubquestion.py: extract sub-questions via AI from guides/transcripts.

Raw data saved as .json or .txt in root or vector_index folder.

3. Build Vector Index
Run indexing scripts in vector_index/ (or via backend_langgraph).

Populate vector store for semantic retrieval.

4. Query Interface
backend.py: process queries through your preferred backend and vector index.

frontend/: serve a simple UI (likely with Flask or FastAPI + JS) that leverages backend API endpoints.

5. Testing & Validation
validation.py: enforce input/output constraints.

test.py: simple test harness; expand with pytest for full coverage.

🧪 Running the Front‑end
Inside /frontend, review the framework (Flask, FastAPI, etc.) and run accordingly:

bash
Copy
Edit
cd frontend
pip install -r ../requirements.txt
python app.py  # or npm/yarn if JS-based
Visit http://localhost:<port> to ask questions, search products, or view guides.
