import os
import pymongo
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables from .env file
load_dotenv(find_dotenv())

# MongoDB Atlas connection details
ATLAS_URI = os.getenv('ATLAS_URI')
DB_NAME = 'sample_products'
COLLECTION_NAME = 'embedded_products'

# Load the dataset (assuming it's a CSV file)
dataset_path = 'path/to/Reviews.csv'  # Update this path to your downloaded dataset
df = pd.read_csv(dataset_path)

# Add image URLs (dummy example, update with actual URLs)
image_urls = [
    'https://via.placeholder.com/300.png/09f/fff',
    'https://www.gstatic.com/webp/gallery3/1.sm.png',
    # ... more URLs corresponding to each product
]

# Initialize the Hugging Face model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Connect to MongoDB Atlas
client = pymongo.MongoClient(ATLAS_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Process and insert product data with embeddings and image URLs
for idx, row in df.iterrows():
    product = {
        "name": row["ProductName"],
        "description": row["Text"],
        "features": row["Summary"].split(', '),
        "embedding_vector": generate_embedding(row["Text"]),
        "image_url": image_urls[idx] if idx < len(image_urls) else None
    }
    collection.insert_one(product)

print("Product data inserted successfully!")
