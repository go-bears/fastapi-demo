"""
This file contains the model loading and tokenization logic for the AYA-35B model tokenizer and generative models.

It is meant to be imported and used by the main.py file.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from loguru import logger
import torch

# Load AYA-35B model and tokenizer
MODEL_NAME = "CohereForAI/aya-23-35b"


try:
    logger.info(f"CUDA availability: {torch.cuda.is_available()}")
    logger.info(f"CUDA device: {torch.cuda.current_device()}")
except:
    logger.error(f"Error checking CUDA availability")


def load_cuda():
    try:
        torch.cuda.is_available()
        cuda_device = torch.cuda.current_device()
        logger.info(f"available cuda device: {torch.cuda.current_device()}")
        return cuda_device
    except:
        logger.info("model loaded on cpu")
    

def load_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        cuda_device = load_cuda()
        if cuda_device:
            model = model.to(cuda_device)

        logger.info(f"Successfully loaded {MODEL_NAME} model and tokenizer")
        return tokenizer, model
    
    except Exception as e:
        logger.error(f"Failed to load {MODEL_NAME} model and tokenizer: {str(e)}")
        raise

if __name__ == "__main__":
    """Bootstrapped testing by running `python model.py`
    - load model and tokenizer
    - generate response
    - decode response
    Sample return should look like:
    
    <|START_OF_TURN_TOKEN|><|USER_TOKEN|>How do you say Hello in Spanish?<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Hello in Spanish is Hola.
    """
    tokenizer, model = load_models()
    messages = [{"role": "user", "content": "How do you say Hello in Spanish?"}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    gen_tokens = model.generate(
        input_ids, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.3,
        )

    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    logger.info(gen_text)
    logger.info("model loaded") 