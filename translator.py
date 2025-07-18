from transformers import MarianMTModel, MarianTokenizer

def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, tokenizer, model):
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translated = model.generate(**tokens)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    model_name = 'Helsinki-NLP/opus-mt-en-hi'
    tokenizer, model = load_model(model_name)

    while True:
        text = input("Enter English text (or type 'exit'): ")
        if text.lower() == 'exit':
            break
        translated = translate(text, tokenizer, model)
        print("Translated to Hindi:", translated)
