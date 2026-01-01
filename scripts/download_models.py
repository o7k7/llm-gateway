from sentence_transformers import SentenceTransformer
from spacy.cli import download

print("Downloading Sentence Transformer...")

SentenceTransformer('all-MiniLM-L6-v2') # Tensor dimension 384

print('Downloading SpaCy models...')

download("en_core_web_lg")

print("Injected images into docker image")