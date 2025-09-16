from flask import Flask, request, render_template, jsonify
import json

from flask_cors import CORS

from transformers import AutoTokenizer
from transformers import BlenderbotForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM


app = Flask(__name__)
CORS(app)

mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = AutoTokenizer.from_pretrained(mname)

conversation_history = []

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json()
        input_text = data.get('prompt', '').strip()
        if not input_text:
            return jsonify({'error': 'Prompt vazio.'}), 400

        # print(f"INPUT TEXT: {input_text}")  # DEBUG

        # Construir histórico limitado (últimas 6 interações)
        history_text = f" {tokenizer.eos_token} ".join(conversation_history[-6:])
        full_input = (history_text + f" {tokenizer.eos_token} " + input_text).strip() if history_text else input_text
        # print(f"FULL INPUT: {full_input}")

        # Tokenizar
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True)

        # Gerar resposta
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            pad_token_id=model.config.pad_token_id
        )

        if outputs is None or len(outputs) == 0:
            response = "Desculpe, não consegui gerar uma resposta."
        else:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Atualizar histórico
        conversation_history.append(input_text)
        conversation_history.append(response)

        return jsonify({'response': response})

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return jsonify({"error": "Erro interno no servidor."}), 500

# def handle_prompt():
#     data = request.get_data(as_text=True)
#     data = json.loads(data)
#     input_text = data['prompt']
#     # Create conversation history string
#     history = "\n".join(conversation_history)
#     # Tokenize the input text and history
#     inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")
#     # Generate the response from the model
#     outputs = model.generate(**inputs, max_length= 60)  # max_length will cause the model to crash at some point as history grows
#     # Decode the response
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     # Add interaction to conversation history
#     conversation_history.append(input_text)
#     conversation_history.append(response)
#     return response


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()