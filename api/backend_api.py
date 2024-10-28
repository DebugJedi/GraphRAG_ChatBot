from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
from app.graphrag import GraphRAG
from pypdf import PdfReader
import numpy as np
app = FastAPI()

# #in memory processed document
# processed_documents = {}

# @app.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile):
#     try:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(await file.read())
#             tmp_file_path = tmp_file.name

#         loader = PyPDFLoader(tmp_file_path)
#         documents = loader.load()

#         global processed_documents
#         processed_documents[file.filename] = documents[:10]

#         return {"message": "PDF processed successfully", "filename": file.filename}

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
    

# @app.post("/ask_query/")
# async def ask_query(query: str = Form(), filename: str= Form()):
#     """Ask a query based on the processed document."""
#     try:
#         if filename not in processed_documents:
#             return JSONResponse(status_code=400, content={"error": "PDF not processed."})

#         graph_rag = GraphRAG()
#         graph_rag.process_documents(processed_documents[filename])

#         output = graph_rag.query(query)

#         if output:
#             response_text = getattr(output, 'content', None) or getattr(output, 'text', None) or str(output)
#             return {"response": response_text}
        
#         return {"response": "No answer found."}
    
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

