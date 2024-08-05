"""
    Fimio Demo of Translation API with AYA-35B model

    This demo provides and REST API for translating text using the AYA-35B model.

    How to test:
    run server with main.py
    `python main.py`
    
    1) check model is running
    - open browser to http://localhost:8080/ which should display sample translation
    
    2) test new translations with curl
    - curl -X POST "http://localhost:8080/translate"  -H "Content-Type: application/json"  -d '{ "role": "user", "content": "how do you say Hello World! in French and Japanese and Turkish" }'

    example response may look like:
    {"role":"assistant","content":"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>how do you say Hello World! in French and Japanese and Turkish<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Here are the translations of \"Hello World!\" in French, Japanese, and Turkish:\n\n- French: \"Bonjour le Monde !\"\n- Japanese: \"世界こんにちは\" (sekaikon'nichiwa)\n- Turkish: \"Merhaba Dünya!\"<|END_OF_TURN_TOKEN|>","time_to_generate":117.22399687767029,"num_generated_tokens":79,"num_input_tokens":19}
    
    3) test /translate endpoint with swagger UI
    - open browser to http://localhost:8080/docs
    - update the request body under 'content'with new text to translate
    press `execute`
    - for example: 
        {
        "content": "how do you say thank you in Spanish, Chinese, and Korean",
        "role": "user"
        }
    - example output: {
        "role": "assistant",
        "content": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>how do you say thank you in Spanish, Chinese, and Korean<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Thank you in Spanish is \"gracias,\" in Chinese is \"谢谢\" (xìe xiè), and in Korean is \"감사합니다\" (gam-sa-hab-ni-da).",
        "time_to_generate": 75.68138647079468,
        "num_generated_tokens": 64,
        "num_input_tokens": 19
        }

    Currently known caveats 7/7/2024 (MF):
    - model is not yet optimized for long outputs
    - model not yet run on GPU

"""

import time

import torch
import uvicorn

from typing import List

from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from langcodes import Languages 
from logging_config import setup_logging
from model import load_models, generate_response


setup_logging()

TOKENIZER, MODEL = load_models()


# Initialize FastAPI app
app = FastAPI(
    title="AYA-35B Translation API",
    description="API for translating text using the AYA-35B model",
)

languages = Languages()

# Pydantic models
class TranslationRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000, description="text input to model")
    role: str = "user"


class TranslationResponse(BaseModel):
    role: str = "assistant"
    content: str
    time_to_generate: float
    num_generated_tokens: int
    num_input_tokens: int

    

class LanguageInfo(BaseModel):
    code: str
    name: str


# Function to get supported languages
def get_supported_languages() -> List[LanguageInfo]:
    return [LanguageInfo(code=code, name=name) for code, name in languages.languages.items()]

# Dependency for languages
async def get_languages():
    return get_supported_languages()

@app.get("/")
async def home():
    logger.info("setting up home endpoint. models are loading...")
    tokenizer = TOKENIZER
    model = MODEL
    # Format message with the command-r-plus chat template
    logger.info("generating initial test translation...")
    messages = [{"role": "user", "content": "how do I say test in Japanese and French?"}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    gen_text, num_output_tokens, time_to_generate = generate_response(input_ids, model=model, tokenizer=tokenizer)

    return {
        "message": "Welcome to the AYA-35B Translation API. This is a inital translation test. Use the /translate endpoint for text translations.",
        "testing": "testing model here",
        "time_taken": str(time_to_generate),
        "device": str(model.device),
        "gen_text": str(gen_text)
    }

@app.get("/languages")
async def get_supported_languages_endpoint(supported_languages: List[LanguageInfo] = Depends(get_languages)):
       return {
           "message": "Here are the supported 23 languages by Aya 2:",
           "supported_languages": supported_languages
       }

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    tokenizer = TOKENIZER
    model = MODEL
    logger.info(f"received: {request}")

    try:
        # Prepare input for the model with tokenizer
        logger.info(f"processing input text: {request.content}")
        messages = [
            {"role": "user", "content": request.content}
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        num_input_tokens = input_ids.shape[1]
        gen_text, num_output_tokens, time_to_generate = generate_response(input_ids, model=model, tokenizer=tokenizer)

        response =  TranslationResponse(
            role="assistant",
            content=gen_text,
            time_to_generate=float(time_to_generate),
            num_generated_tokens=num_output_tokens,
            num_input_tokens=num_input_tokens
        )
        logger.info(f"response: {response}")
        return response

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during translation")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
