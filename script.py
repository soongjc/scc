from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
model.save("./models/multi-qa-mpnet-base-dot-v1")