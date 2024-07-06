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

    Currently known caveats 7/6/2024 (MF):
    - model is not yet optimized for multi-language translations
    - model is not yet optimized for long outputs
    - model not yet run on GPU

"""

import time

import torch
import uvicorn

from typing import List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_config import logger
from langcodes import Languages 
from model import load_models

from logging_config import setup_logging

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
    tokenizer = TOKENIZER
    model = MODEL
    # Format message with the command-r-plus chat template
    messages = [{"role": "user", "content": "how do I say test in Japanese and French?"}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

     # Generate translation
    logger.info("generating translation")
    start_time = time.time()
    logger.info(f"generative model device: {model.device}")
    gen_tokens = model.generate(
        input_ids, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.3,
        )

    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    end_time = time.time()
    logger.info(f"time taken: {str(end_time - start_time)}")
    logger.info(f"outputs: {str(gen_text)}")

    return {
        "message": "Welcome to the AYA-35B Translation API. This is a inital translation test. Use the /translate endpoint for text translations.",
        "testing": "testing model here",
        "time_taken": str(end_time - start_time),
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
        model = model.to(torch.cuda.current_device())
        logger.info(f"model is on cuda device: {model.device}")
    except:
        logger.error(f"Error moving model to cuda device. model device is: {model.device}")
    try:
        # Prepare input for the model with tokenizer
        logger.info(f"processing input text: {request.content}")
        message = [
            {"role": "user", "content": request.content}
        ]
        input_ids = tokenizer.apply_chat_template(message,
        add_generation_prompt=True,return_tensors="pt")


        # Generate translation
        logger.info("generating translation")
        logger.info(f"model device: {model.device}")
        start_time = time.time()
        
        gen_tokens = model.generate(
        input_ids, 
            max_new_tokens=300, 
            do_sample=True, 
            temperature=0.3,
            )

        gen_text = str(tokenizer.decode(gen_tokens[0]))
        logger.info(f"generated text: {gen_text}")
        end_time = time.time()
        
        logger.info(f"time taken: {str(end_time - start_time)}")
        logger.info(f"Successfully processed text.")


        return TranslationResponse(
            role="assistant",
            content=gen_text,
            time_to_generate=float(end_time - start_time),
            num_generated_tokens=int(gen_tokens.shape[1]),
            num_input_tokens=int(input_ids.shape[1])
        )

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during translation")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
