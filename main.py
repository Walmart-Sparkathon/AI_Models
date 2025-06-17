from fastapi import FastAPI, Request
from pydantic import BaseModel
import spacy
from fastapi.middleware.cors import CORSMiddleware

nlp = spacy.load("./product_filter_nlp")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str
    
@app.post("/analyze")
def analyze_text(request: QueryRequest):
    doc = nlp(request.message)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return {"entities": entities}