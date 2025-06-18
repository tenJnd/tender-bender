from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from src.documents_parser import ParsedDocumentData


class DocumentSearchEngine:
    def __init__(self, documents_metadata: List[ParsedDocumentData],
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 chunk_size: int = 800, overlap: int = 200):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._prepare_chunks(documents_metadata)
        self._build_index()

    def _prepare_chunks(self, documents_metadata: List[ParsedDocumentData]):
        for doc in documents_metadata:
            doc_name = doc.name
            full_text = doc.full_text
            if not full_text:
                continue

            start = 0
            while start < len(full_text):
                end = min(start + self.chunk_size, len(full_text))
                chunk = full_text[start:end].strip()
                if len(chunk) > 100:
                    self.chunks.append(chunk)
                    self.chunk_metadata.append({
                        "document": doc_name,
                        "start": start,
                        "end": end
                    })
                start += self.chunk_size - self.overlap

    def _build_index(self):
        embeddings = self.model.encode(self.chunks, convert_to_tensor=True, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.cpu().numpy())
        self.embeddings = embeddings

    def query(self, question: str, k: int = 5) -> List[dict]:
        question_embedding = self.model.encode([question])
        D, I = self.index.search(question_embedding, k)
        return [
            {
                "text": self.chunks[idx],
                "metadata": self.chunk_metadata[idx]
            }
            for idx in I[0]
        ]
