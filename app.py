import os
import json
import faiss

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import streamlit as st

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

@st.cache_resource
def load_index():
    return faiss.read_index(
        "vectorstore/ramayana.index"
    )

@st.cache_data
def load_documents():

    docs = []

    with open(
        "translated/documents.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            docs.append(
                json.loads(line)
            )

    return docs

embedding_model = load_model()
index = load_index()
documents = load_documents()

SYSTEM_PROMPT = """
You are a Ramayana expert assistant.

Answer ONLY from the provided context.

If the answer is not found, reply:

I could not find this information in the Ramayana knowledge base.

Provide concise and accurate answers.
"""

st.set_page_config(
    page_title="Ramayana AI Assistant",
    page_icon="🕉️",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color:#FFF8E7;
}

.stTextInput input{
    font-size:18px;
}

.answer-box{
    padding:20px;
    border-radius:15px;
    background:#fff;
    border:2px solid #FFB74D;
}

</style>
""", unsafe_allow_html=True)

st.title("🕉️ Ramayana AI Knowledge Assistant")

st.markdown(
"""
Ask questions from:

- Bala Kanda
- Ayodhya Kanda
- Aranya Kanda
- Kishkindha Kanda
- Sundara Kanda
- Yuddha Kanda
"""
)

question = st.text_input(
    "Ask your question"
)

if st.button("🔍 Ask"):

    if question:

        query_embedding = embedding_model.encode(
            [question],
            normalize_embeddings=True
        )

        scores, indices = index.search(
            query_embedding.astype("float32"),
            k=5
        )

        context_parts = []

        retrieved = []

        for idx in indices[0]:

            doc = documents[idx]

            retrieved.append(
                f"{doc['metadata']['kanda']} - {doc['metadata']['chapter_ordinal']}"
            )

            context_parts.append(
                doc["text"]
            )

        context = "\n\n".join(
            context_parts
        )

        user_prompt = f"""
Context:

{context}

Question:

{question}

Answer:
"""

        with st.spinner(
            "Consulting the Ramayana..."
        ):

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role":"system",
                        "content":SYSTEM_PROMPT
                    },
                    {
                        "role":"user",
                        "content":user_prompt
                    }
                ],
                temperature=0
            )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        st.subheader(
            "📜 Retrieved Chapters"
        )

        for item in retrieved:
            st.write("•", item)

        st.subheader(
            "🪔 AI Answer"
        )

        st.success(answer)