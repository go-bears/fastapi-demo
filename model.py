"""
This file contains the model loading and tokenization logic for the AYA-35B model tokenizer and generative models.

It is meant to be imported and used by the main.py file.
"""
import time

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from loguru import logger


# Load AYA-35B model and tokenizer
# MODEL_NAME = "CohereForAI/aya-23-35b"
MODEL_NAME = "CohereForAI/aya-23-8B"

def load_cuda():
    """Load CUDA if available"""
    device = None
    try:
        logger.info(f"CUDA availability: {torch.cuda.is_available()}")
        logger.info(f"CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = "cuda"
    except:
        logger.info("cannot access cuda")

    return device

    

def load_models() -> tuple:
    """Load model and tokenizer from Hugging Face model hub"""
    device = None
    try:
        device = load_cuda()
        if torch.cuda.device_count() > 1:
            ## device_map option requires HF's acclerator library is installed ## and will parallelize work across multiple GPUs when set to auto
            logger.info(f"{torch.cuda.device_count()} GPUs available, setting device to auto")
            device = 'auto'

    except Exception as e:
        device = 'cpu'
        logger.error(f"Cannot load cuda to {MODEL_NAME} model or tokenizer: {str(e)}. device set to CPU" )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map = device  )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map = device)

    logger.info(f"Successfully loaded {MODEL_NAME} model  on {model.device}")

    return tokenizer, model


def generate_response(input_ids, model, tokenizer) -> tuple:
    """Generate response from model
    
    Args:
    - messages (list): list of dictionaries containing the role and content of the message
    - input_ids (torch.Tensor): tensor of input_ids
    - model (transformers.PreTrainedModel): model to generate response
    - tokenizer (transformers.PreTrainedTokenizer): tokenizer to encode and decode text
    
    Returns a tuple of:
    - gen_text (str): generated text
    - num_gen_tokens (int): number of generated tokens
    - time_to_generate (float): time taken to generate response
    """

     # Generate translation
    logger.info("generating translation")
    start_time = time.time()
    logger.info(f"generative model device: {model.device}")
    input_ids = input_ids.to(model.device)
    gen_tokens = model.generate(
        input_ids, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.3,
        )

    num_gen_tokens = gen_tokens.shape[1]
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    end_time = time.time()
    time_to_generate = end_time - start_time
    
    logger.info(f"time taken: {time_to_generate}")
    logger.info(f"num output tokens: {num_gen_tokens}")
    logger.info(f"generation time: {time_to_generate}")

    return gen_text, num_gen_tokens, time_to_generate

    

if __name__ == "__main__":
    """Bootstrapped testing by running `python model.py`
    - load model and tokenizer
    - generate response
    - decode response
    Sample return should look something like:
    
    <|START_OF_TURN_TOKEN|><|USER_TOKEN|>How do you say Hello in Spanish?<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Hello in Spanish is Hola.
    """
    tokenizer, model = load_models()
    
    messages = [{"role": "user", "content": "How do you say Hello in Spanish?"}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    gen_text, num_gen_tokens, time_to_generate = generate_response(input_ids, model, tokenizer)
    print(num_gen_tokens, "---", gen_text, "---", time_to_generate)
