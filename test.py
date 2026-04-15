from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
import sacrebleu
import torch
import numpy as np


base_model_name = "Qwen/Qwen2.5-0.5B"  # samakan dengan train.py
lora_path = "./model-lora"
device = "cpu"

prompts = [
    "Buatkan satu soal cerita matematika berbahasa Indonesia untuk siswa kelas 2 SD tentang penjumlahan.",
    "Buatkan satu soal cerita matematika berbahasa Indonesia untuk siswa kelas 3 SD tentang pengurangan.",
    "Buatkan satu soal cerita matematika berbahasa Indonesia untuk siswa kelas 4 SD tentang uang.",
    "Buatkan satu soal cerita matematika berbahasa Indonesia untuk siswa kelas 2 SD tentang waktu.",
    "Buatkan satu soal cerita matematika berbahasa Indonesia untuk siswa kelas 3 SD tentang perkalian.",
    "Buatkan satu soal cerita matematika berbahasa Indonesia untuk siswa kelas 4 SD tentang panjang.",
    "Buatkan satu soal cerita matematika berbahasa Indonesia untuk siswa kelas 5 SD tentang desimal.",
]


tokenizer = AutoTokenizer.from_pretrained(lora_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(base_model, lora_path)
model.to(device)
model.eval()


generated_texts = []

for i, prompt in enumerate(prompts, 1):
    full_prompt = f"Instruction: {prompt}\nResponse:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Response:" in decoded:
        result = decoded.split("Response:", 1)[1].strip()
    else:
        result = decoded.strip()

    generated_texts.append(result)

    print(f"\n=== Soal {i} ===")
    print(result)


def self_bleu(texts):
    if len(texts) < 2:
        return 0.0

    scores = []
    for i, text in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        # format refs untuk sacrebleu: list of reference sets
        bleu = sacrebleu.corpus_bleu([text], [refs])
        scores.append(bleu.score)
    return sum(scores) / len(scores)


def jaccard_similarity(texts):
    if len(texts) < 2:
        return 0.0

    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts).toarray()

    scores = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            score = jaccard_score(X[i], X[j], average="binary")
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def cosine_similarity(texts):
    if len(texts) < 2:
        return 0.0

    embeddings = embedder.encode(texts, normalize_embeddings=True)

    scores = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            scores.append(float(np.dot(embeddings[i], embeddings[j])))

    return float(np.mean(scores)) if scores else 0.0


print("\n=== HASIL EVALUASI ===")
print("Self-BLEU:", self_bleu(generated_texts))
print("Jaccard:", jaccard_similarity(generated_texts))
print("Cosine:", cosine_similarity(generated_texts))