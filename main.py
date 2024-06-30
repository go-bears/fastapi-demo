import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AYA-35B Translation API",
    description="API for translating text using the AYA-35B model",
)

# Load AYA-35B model and tokenizer
MODEL_NAME = "CohereForAI/aya-23-35B"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    logger.info(f"Successfully loaded {MODEL_NAME} model and tokenizer")
except Exception as e:
    logger.error(f"Failed to load {MODEL_NAME} model and tokenizer: {str(e)}")
    raise


# Pydantic model for translation request
class TranslationRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=1000, description="The text to be translated"
    )
    target_language: str = Field(
        ...,
        min_length=2,
        max_length=5,
        description="The target language code (e.g., 'fr' for French)",
    )
    source_language: Optional[str] = Field(
        None,
        min_length=2,
        max_length=5,
        description="The source language code (optional)",
    )


# Pydantic model for translation response
class TranslationResponse(BaseModel):
    translated_text: str
    source_language: Optional[str]
    target_language: str


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        # Prepare input for the model
        input_text = f"Translate to {request.target_language}: {request.text}"
        if request.source_language:
            input_text = f"Translate from {request.source_language} to {request.target_language}: {request.text}"

        # Tokenize input
        inputs = tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        )

        # Generate translation
        outputs = model.generate(
            **inputs, max_length=512, num_return_sequences=1, do_sample=True
        )

        # Decode the generated output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"Successfully translated text to {request.target_language}")
        return TranslationResponse(
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
        )

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred during translation"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
