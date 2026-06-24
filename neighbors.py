from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def get_neighbors(word):
    embeddings = model.transformer.wte.weight
    word_id = tokenizer.encode(word)[0]
    word_vector = embeddings[word_id]
    
    similarities = F.cosine_similarity(word_vector, embeddings, dim=1)

    top_ids = torch.topk(similarities, 6)

    top_words = []
    
    for token_id in top_ids.indices:
        word = tokenizer.decode(token_id)
        top_words.append(word)
    
    return top_words


word = input("Enter word: ")
words = get_neighbors(word)
for w in words:
    print(w)