from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

app = FastAPI()

# Load smaller model and tokenizer...
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    inputs = tokenizer(request.text, max_length=1024, truncation=True, return_tensors="tf")
    
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=request.max_length,
        min_length=request.min_length,
        length_penalty=2.0,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)