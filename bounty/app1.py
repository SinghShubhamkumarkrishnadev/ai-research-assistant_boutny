# Import libraries
import streamlit as st
from dotenv import find_dotenv, dotenv_values
import pymongo

# Load environment variables
config = dotenv_values(find_dotenv())
ATLAS_URI = config.get('ATLAS_URI')
DB_NAME = 'your_database_name'
COLLECTION_NAME = 'your_collection_name'

# Initialize MongoDB connection
client = pymongo.MongoClient(ATLAS_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Page title and description
st.title("🔍 AI Research Assistant")
st.markdown("Welcome to the AI Research Assistant! Search for research papers in any field.")

# User input: Research query
research_query = st.text_input("🔎 What topic are you researching?")

# Button to trigger search
if st.button("🚀 Get Recommendations"):
    # Retrieve papers based on query
    papers = collection.find({"$text": {"$search": research_query}}).limit(5)
    
    # Display search results
    if papers.count() > 0:
        st.markdown(f"### 📚 Found {papers.count()} papers related to '{research_query}':")
        st.write("")

        for idx, paper in enumerate(papers):
            st.markdown(f"**Paper {idx+1}:**")
            st.write(f"**Title:** {paper['title']}")
            st.write(f"**Authors:** {paper['authors']}")
            st.write(f"**Abstract:** {paper['abstract']}")
            st.write(f"**Link:** [{paper['link']}]({paper['link']})")
            st.write("")
    else:
        st.warning("No papers found. Please try a different query.")
