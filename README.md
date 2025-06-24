🌟 TerShine
A Python-based toolkit for scraping, vector indexing, and querying product and guide data. Ideal for building question-answering tools or recommendation systems utilizing both structured and unstructured data.

🚀 Features
Web scraping of product listings, washing guides, and transcripts

Vector indexing for retrieving semantically relevant information

Multiple backends supported: basic, LangGraph, and custom logic

Validation and testing functionality for development

Front‑end component for interactive querying

📂 Repository Structure
bash
Copy
Edit
tershine/
├── backend.py                # Main backend logic
├── backend_langgraph.py      # Backend alternative experimentation using LangGraph
├── aiscrapesubquestion.py    # Testing sub-question decomposition from big question
├── extract_wg_links.py       # Extract washing guide URLs from Tershine webpage
├── get_products.py           # Scrape raw product data
├── validation.py             # For testing purposes, can ignore
├── test.py                   # [Incomplete] Experiment on fine-tuning AI model on voice data
├── requirements.txt          # Python dependencies
├── vector_index/             # Vector index data and helper code
├── frontend/                 # Web UI and assets for interaction
├── *.txt / *.json / *.pyc    # Raw data dumps, transcripts, caches
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
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🔧 Usage
1. Scrape Products & Guides
get_products.py: Identify and save product links/data

extract_wg_links.py: Extract washing guide URLs from Tershine website

2. Scrape & Process Datasets
aiscrapesubquestion.py: Extract sub-questions via AI from guides/transcripts

Output is saved as .json or .txt in root or vector_index/

3. Build Vector Index
Run indexing scripts in vector_index/ or via backend_langgraph.py

This populates the vector store for semantic retrieval

4. Query Interface
backend.py: Main backend logic using your preferred method

frontend/: Serve the UI (Flask/FastAPI or JS) that queries the backend

5. Testing & Validation
validation.py: Schema and constraint checks (optional)

test.py: Initial testing setup (expand using pytest)

🧪 Running the Frontend
Inside the frontend/ folder:

bash
Copy
Edit
cd frontend
pip install -r ../requirements.txt
python app.py  # or use npm/yarn if frontend is JS-based
Visit http://localhost:PORT in your browser to interact with the system.
