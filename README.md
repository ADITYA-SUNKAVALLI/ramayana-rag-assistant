# Ramayana RAG Knowledge Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system built from a Telugu Ramayana Question-Answer book.

The system extracts text from PDF, cleans OCR errors, parses chapters, translates Telugu content into English using DeepSeek, generates embeddings, stores them in FAISS, and provides an interactive AI chatbot interface using Streamlit.

---

## Features

- PDF Text Extraction
- Telugu OCR Cleanup
- Chapter & Q/A Parsing
- DeepSeek Translation
- Semantic Search using FAISS
- Retrieval-Augmented Generation (RAG)
- Streamlit Chat Interface
- Explainable Retrieval (Source Chapters)

---

## Project Pipeline

```text
Ramayana PDF
      │
      ▼
01_extract.py
      │
      ▼
Extracted Text Batches
      │
      ▼
01b_merge_batches.py
      │
      ▼
full_corpus.txt
      │
      ▼
02_clean_and_parse.py
      │
      ▼
all_chapters.jsonl
      │
      ▼
fix_common_ocr.py
      │
      ▼
all_chapters_fixed.jsonl
      │
      ▼
03_translate.py
      │
      ▼
all_chapters_en.jsonl
      │
      ▼
04_strip_for_rag.py
      │
      ▼
rag_final_chapters.jsonl
      │
      ▼
05_prepare_documents.py
      │
      ▼
documents.jsonl
      │
      ▼
06_generate_embeddings.py
      │
      ▼
embeddings.npy
metadata.json
      │
      ▼
07_build_faiss.py
      │
      ▼
FAISS Vector Store
      │
      ▼
09_rag_chat.py
      │
      ▼
Streamlit Application
```

---

## Dataset

Source:

Telugu Ramayana Question & Answer Book

Content:

- 99 Chapters
- 6 Kandas
  - Bala Kanda
  - Ayodhya Kanda
  - Aranya Kanda
  - Kishkindha Kanda
  - Sundara Kanda
  - Yuddha Kanda

---

## Folder Structure

```text
RAMAYANA-RAG-PIPELINE/

├── raw_pdf/
├── extracted_text/
├── cleaned_text/
├── translated/
├── embeddings/
├── vectorstore/
├── logs/
├── scripts/
├── app.py
├── requirements.txt
├── README.md
└── .env
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/ADITYA-SUNKAVALLI/ramayana-rag-assistant.git

cd ramayana-rag-assistant
```

---

### Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create `.env`

```env
DEEPSEEK_API_KEY=your_api_key_here
```

---

## Running Pipeline

### Step 1

Extract PDF

```bash
python scripts/01_extract.py
```

### Step 2

Merge Batches

```bash
python scripts/01b_merge_batches.py
```

### Step 3

Clean and Parse

```bash
python scripts/02_clean_and_parse.py
```

### Step 4

Cleaning OCR

```bash
python scripts/fix_common_ocr.py


### Step 5

Translate

```bash
python scripts/03_translate.py
```

### Step 6

Rag final chapters`

```bash
python scripts/04_strip_for_rag.py

### Step 7

Prepare Documents

```bash
python scripts/05_prepare_documents.py
```

### Step 8

Generate Embeddings

```bash
python scripts/06_generate_embeddings.py
```

### Step 9

Build FAISS Index

```bash
python scripts/07_build_faiss.py
```

---

## Run Chatbot

Terminal Chat:

```bash
python scripts/09_rag.py
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

## Example Questions

- Who first propagated the Ramayana?
- Who killed Tataka?
- Why did Bharata refuse the throne?
- Describe Sita's marriage story.
- Why did Vibhishana leave Ravana?
- How did Rama cross the ocean?

---

## Technologies Used

- Python
- DeepSeek API
- FAISS
- Sentence Transformers
- Streamlit
- PyMuPDF
- NumPy
- JSONL

---

## Author

Aditya

B.Tech Computer Science

AI Engineering / RAG Project
