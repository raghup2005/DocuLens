from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil

from app.main import RAGPipeline
from backend.services import ingest_document


app = FastAPI(
    title="Multimodal RAG API",
    description="Backend API for the Multimodal RAG system",
    version="1.0.0"
)


rag_pipeline = RAGPipeline()


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    question: str
    search_type: str = "all"
    figure_number: int | None = None


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "Multimodal RAG backend is running"
    }


# ============================================================
# QUERY
# ============================================================

@app.post("/query")
def query_rag(request: QueryRequest):

    content_type = None

    if request.search_type in ["text", "table", "image"]:
        content_type = request.search_type

    try:

        answer = rag_pipeline.ask(
            query=request.question,
            content_type=content_type,
            figure_number=request.figure_number
        )

        return {
            "status": "success",
            "question": request.question,
            "search_type": request.search_type,
            "figure_number": request.figure_number,
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# DOCUMENT INGESTION
# ============================================================

@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...)
):

    try:

        raw_dir = Path("data/raw")

        raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path = raw_dir / file.filename

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"\nUploaded file: {file_path}"
        )

        result = ingest_document(
            str(file_path)
        )

        return {
            "status": "success",
            "message": "Document ingested successfully",
            "file": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    