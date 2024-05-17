import streamlit as st
from dotenv import find_dotenv, dotenv_values
import pymongo
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import requests
from io import BytesIO

# Load environment variables from .env file
config = dotenv_values(find_dotenv())
ATLAS_URI = config.get('ATLAS_URI')
DB_NAME = 'sample_products'
COLLECTION_NAME = 'embedded_products'
EMBEDDING_MODELS = {
    'sentence-transformers/all-MiniLM-L6-v2': 'embedding_vector'
}

# Initialize MongoDB Atlas connection
client = pymongo.MongoClient(ATLAS_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Initialize Hugging Face model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def run_vector_query(query, model_name):
    embedding_attr = EMBEDDING_MODELS.get(model_name)
    query_embedding = generate_embedding(query)

    pipeline = [
        {
            "$search": {
                "index": embedding_attr,
                "knnBeta": {
                    "vector": query_embedding,
                    "path": embedding_attr,
                    "k": 5
                }
            }
        }
    ]

    products = collection.aggregate(pipeline)
    return list(products)

# Streamlit App
st.set_page_config(page_title="Product Recommendation System", layout="wide")
st.title("🌟 Product Recommendation System 🌟")

#Adjust image
banner_image = Image.open("banner-image.png")
banner_image = banner_image.resize((600, 500))  # Adjust the size as needed
st.image(banner_image, use_column_width=False)

# User Input: Search Query
user_query = st.text_input("🔍 What product are you looking for?", "")

# User Input: Model Selection
selected_model = st.selectbox("🧠 Choose an embedding model:", list(EMBEDDING_MODELS.keys()))

# Button to trigger the recommendation
if st.button("💡 Recommend Products"):
    if user_query:
        with st.spinner("🔄 Searching for products..."):
            recommended_products = run_vector_query(user_query, selected_model)
        st.success(f"🎉 Found {len(recommended_products)} products:")
        
        # Display recommended products
        for idx, product in enumerate(recommended_products):
            st.markdown(f"### {idx + 1}. {product['name']}")
            st.markdown(f"**Description:** {product['description']}")
            st.markdown(f"**Features:** {', '.join(product['features'])}")
            
            # Add product image if available
            if "image_url" in product:
                response = requests.get(product["image_url"])
                img = Image.open(BytesIO(response.content))
                st.image(img, width=150)
            st.markdown("---")
    else:
        st.warning("⚠️ Please enter a product query.")
