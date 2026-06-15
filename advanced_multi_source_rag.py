# ===============================================================
# ADVANCES MULTI SOURCE RAG
# ================================================================

# ========================================================================================
# STEP 1: IMPORTING OF FILES OR DOCUMENTS by using langchain_community.documents_loaders
# =======================================================================================

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)

import pandas as pd

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI)

from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader


from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline

# ==================================================================================================
# STEP 2 LOAD PDF
# =================================================================================================

pdf_loader = PyPDFLoader("data/HR_Policy.pdf")

pdf_docs = pdf_loader.load()

print("PDF DOCUMENTS:", len(pdf_docs))

# ==================================================================================================
# STEP 3 LOAD DOCX
# =================================================================================================

docx_loader = Docx2txtLoader("data/Employee_Handbook.docx")

docx_docs = docx_loader.load()

print("DOCX DOCUMENTS:", len(docx_docs))

# ==================================================================================================
# STEP 4 LOAD CSV
# =================================================================================================

csv_loader = CSVLoader("data/Annual_Report.csv")

csv_docs = csv_loader.load()

print("CSV DOCUMENTS:", len(csv_docs))

# ==================================================================================================
# STEP 5 LOAD HTML
# =================================================================================================

html_loader = UnstructuredHTMLLoader("data/HR_data.html")

html_docs = html_loader.load()

print("HTML DOCUMENTS:", len(html_docs))

# ==================================================================================================
# STEP 6 LOAD EXCEL
# =================================================================================================

df = pd.read_excel(
    "data/Employee_Master.xlsx",
    header=5
)

df = df.iloc[:, 1:]

print(df.head())
print(df.columns)
print(df.shape)

excel_docs = []

for _, row in df.iterrows():

    excel_docs.append(
        Document(
            page_content=str(row.to_dict())
        )
    )

print("EXCEL DOCUMENTS:", len(excel_docs))

# ==================================================================================================
# STEP 7 LOAD WEBSITE OR URL
# =================================================================================================

web_loader = WebBaseLoader(
    "https://1clickdataanalysis.com/"
)

web_docs = web_loader.load()

print(
    "WEBSITE DOCUMENTS:",
    len(web_docs)
)

# ==================================================================================================
# STEP 8 COMBINING ALL DOCUMENTS
# =================================================================================================

all_documents = pdf_docs + docx_docs + csv_docs + html_docs +  excel_docs + web_docs

print("ALL DOCUMENTS:", len(all_documents))

# ==================================================================================================
# STEP 9 CHUNKING
# =================================================================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(
    all_documents
)

print("TOTAL CHUNKS:", len(chunks))

print(type(chunks))
print(type(chunks[0]))

print("\nFIRST CHUNK:\n")
print(chunks[0].page_content)

# ============================================================================================
# STEP 10 : EMBEDDINGS
# ============================================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Created")

vector = embeddings.embed_query(
    chunks[0].page_content
)

print(type(vector))
print(len(vector))
print(vector[:5])

# ===========================================================================================
# STEP 11: CREATION OF VECTOR DATABASE
# ==========================================================================================

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings
)

print("Vector Store Created")

# =============================================================================================
# STEP 12 : RETRIEVAL
# =============================================================================================

query = input("Enter your query: ")

results = vectorstore.similarity_search(
    query,
    k=3
)

print("\nTop Retrieved Chunks:\n")

for i, doc in enumerate(results, start=1):

    print(f"\n----RESULT {i}------\n")

    print(doc.page_content)

# =================================================================================================
# STEP 13: CREATE CONTEXT
# ==================================================================================================

context = ""

for doc in results:
    context += doc.page_content + "\n\n"

# ====================================================================================================
# STEP 14: GEMINI LLM
# ======================================================================================================

llm = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    max_new_tokens=200
)

print("LLM Created")
# =======================================================================================================
# STEP 15: BUILD PROMPT
# ==========================================================================================================

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
print("Building Prompt")

# =============================================================================================
# STEP 16 : GENERATION
# ===============================================================================================

response = llm(prompt)

print("\n========================")
print("FINAL ANSWER")
print("========================\n")

print(response[0]["generated_text"])

# ========================================================================