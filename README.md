# Aya+fastapi Demo

demo using fastapi to serve multilingual translations with Aya 23-35B model.
You can interact with the model using the chat interface or the /translate endpoint. You can review which languages are available for translation in the model at /languages. Logging is also implemented to track the number of tokens generated and the time it took to generate.

This demo is intended as a functional test of Fimio's features.

### How to use:
1. Clone the repository
2. activate virtual environment with uv
    `uv venv`
3. install requirements
    `uv pip install -r requirements.txt`
4. start the server
    `python main.py`

### how to check model is running

- open browser to http://localhost:8080/ which should display sample translation
    
- test new translations with curl:
    `curl -X POST "http://localhost:8080/translate"  -H "Content-Type: application/json"  -d '{ "role": "user", "content": "how do you say Hello World! in French and Japanese and Turkish" }'`

    the example response may look like:
    ```bash
    {"role":"assistant","content":"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>how do you say Hello World! in French and Japanese and Turkish<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Here are the translations of \"Hello World!\" in French, Japanese, and Turkish:\n\n- French: \"Bonjour le Monde !\"\n- Japanese: \"世界こんにちは\" (sekaikon'nichiwa)\n- Turkish: \"Merhaba Dnya!\"<|END_OF_TURN_TOKEN|>","time_to_generate":117.22399687767029,"num_generated_tokens":79,"num_input_tokens":19}```
    
- test /translate endpoint with swagger UI
    - open browser to http://localhost:8080/docs
    - update the request body under 'content' with new text to translate
    press `execute`

### Currently known caveats 7/6/2024 (MF):
- model is not yet optimized for multi-language translations
- model is not yet optimized for long outputs
- model not yet run on GPU
