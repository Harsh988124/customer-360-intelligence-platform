from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# Create ChromaDB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="customer_business_knowledge"
)


# ---------------------------------------------------------
# Read knowledge documents
# ---------------------------------------------------------

documents = []
metadatas = []
ids = []

doc_number = 0

for file_path in KNOWLEDGE_DIR.glob("*.txt"):

    print(f"Reading: {file_path.name}")

    text = file_path.read_text(
        encoding="utf-8"
    )

    # Simple paragraph-based chunking
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    for chunk_number, chunk in enumerate(chunks):

        documents.append(chunk)

        metadatas.append({
            "source": file_path.name
        })

        ids.append(
            f"doc_{doc_number}_{chunk_number}"
        )

    doc_number += 1


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

print(f"Creating embeddings for {len(documents)} chunks...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
).tolist()


# ---------------------------------------------------------
# Store in ChromaDB
# ---------------------------------------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


print()
print("RAG vector database created successfully.")
print(f"Knowledge chunks: {len(documents)}")
print(f"Database location: {CHROMA_DIR}")