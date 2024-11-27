import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
import ast
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

st.set_page_config(
    page_title="AI Compliance Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize the Google Serper API
search = GoogleSerperAPIWrapper(k=10)

def intro():
    """Introduction page for the assistant."""
    st.write("# Welcome to AI legal Assistant! 📚")
    st.sidebar.success("Select a feature to begin.")
    st.markdown(
        """
        The AI Compliance Assistant is designed to help professionals stay up-to-date with the latest regulations, 
        frameworks, and compliance standards related to AI. It provides insights, generates reports, and answers questions 
        based on available regulatory information.

        ### Features
        - **Crawl and Summarize AI Compliance Information**: Fetch and summarize the latest regulations and frameworks.
        - **Ongoing Updates**: Stay on top of emerging trends in AI compliance.

        **👈 Use the sidebar to select a feature and get started!**
        """
    )

def compliance_report():
    """Fetch and summarize compliance regulations into a report."""
    st.write("# Crawl and Summarize AI Compliance Information")
    
    @st.cache_resource(show_spinner=False)
    def fetch_compliance_data():
        """Crawl compliance and regulatory information."""
        keywords = ["AI compliance frameworks", "AI regulations", "corporate AI policies"]
        result = []
        for keyword in keywords:
            result.append(search.run(keyword))
        return result

    st.write("### Fetching regulatory content, please wait...")
    try:
        compliance_data = fetch_compliance_data()
        st.success("Regulatory data successfully fetched!")
        
        st.write("### Preliminary Regulatory Information")
        for idx, data in enumerate(compliance_data):
            st.markdown(f"**Entry {idx + 1}:**\n\n{data}")
        
        # Summarize the fetched content into a report
        summary_prompt = (
            "Based on the following crawled compliance and regulatory information, generate a concise internal report "
            "including key points, trends, and potential implications:\n" + "\n".join(compliance_data)
        )

        # Prompt the user to input their OpenAI API key
        user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if not user_api_key:
            st.warning("Please provide your OpenAI API key to generate the report.")
            return
        
        client = OpenAI(api_key=user_api_key)
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": "You are an AI compliance expert, summarizing regulatory content."},
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=3000,
            temperature=0.5,
        )
        
        report = response.choices[0].message.content
        st.write("### Generated Internal Report")
        st.markdown(report)
    except Exception as e:
        st.error(f"Error fetching or summarizing regulatory content: {e}")

def compliance_qna():
    """Answer questions based on regulatory content."""
    st.write("# AI Compliance Q&A")

    @st.cache_resource(show_spinner=False)
    def load_prepared_data():
        """Load preprocessed compliance text and embeddings."""
        csv_path = "compliance_data.csv"
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Compliance data file not found: {csv_path}")
        
        texts = df['text'].tolist()
        embeddings = np.vstack([ast.literal_eval(e) for e in df['embedding']])
        
        # Create a Faiss index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return texts, index

    try:
        texts, index = load_prepared_data()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        question = st.text_input("Enter your compliance question:")
        if question:
            query_embedding = model.encode([question])
            _, indices = index.search(query_embedding, k=5)
            
            retrieved_texts = [texts[idx] for idx in indices[0]]
            context = "\n".join(retrieved_texts)
            
            # Prompt the user to input their OpenAI API key
            user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if not user_api_key:
                st.warning("Please provide your OpenAI API key to answer the question.")
                return
            
            client = OpenAI(api_key=user_api_key)
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "You are a compliance assistant answering regulatory questions."},
                    {"role": "user", "content": f"Related content:\n{context}\n\nQuestion: {question}\nAnswer:"},
                ],
                max_tokens=1500,
                temperature=0.5,
            )
            
            answer = response.choices[0].message.content
            st.write("### Q&A Result")
            st.markdown(answer)
    except Exception as e:
        st.error(f"Error loading or answering compliance questions: {e}")

# Mapping pages to their respective functions
page_names_to_funcs = {
    "Introduction": intro,
    "Crawl and Summarize AI Compliance Information": compliance_report,
    #": compliance_qna,
}

# Sidebar page selector
selected_page = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
