import streamlit as st
import google.generativeai as genai
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
import requests
import tempfile
import os
from google.api_core import exceptions
from dotenv import load_dotenv
import time
from urllib.parse import unquote

load_dotenv()

api_key = st.secrets["GEMINI_API_KEY"]
if not api_key:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def analyze_medical_report(content, content_type):
    prompt = """You are an AI medical assistant that answers queries based on the given context and relevant medical knowledge. 
    Here are some guidelines:
    - Prioritize information from the provided documents but supplement with general medical knowledge when necessary.
    - Ensure accuracy, citing sources from the document where applicable.
    - Provide confidence scoring based on probability and reasoning.
    - Be concise, informative, and avoid speculation.
    YOU WILL ANALYSE ONLY MEDICAL DATA, if other CONTEXT is PASSED you will say "Provide Relevant Medical Data. Thanks"
    Answer:
    - **Response:** 
    - **Reasoning:** (explain why this answer is correct and any potential limitations)
"""
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(f"{prompt}\n\n{content}")
            return response.text
        except exceptions.GoogleAPIError as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"An error occurred. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report after {MAX_RETRIES} attempts. Error: {str(e)}")
                return fallback_analysis(content, content_type)

def fallback_analysis(content, content_type):
    st.warning("Using fallback analysis method due to API issues.")
    if content_type == "image":
        return "Unable to analyze the image due to API issues. Please try again later or consult a medical professional for accurate interpretation."
    else:  # text
        word_count = len(content.split())
        return f"""
        Fallback Analysis:
        1. Document Type: Text-based medical report
        2. Word Count: Approximately {word_count} words
        3. Content: The document appears to contain medical information, but detailed analysis is unavailable due to technical issues.
        4. Recommendation: Please review the document manually or consult with a healthcare professional for accurate interpretation.
        5. Note: This is a simplified analysis due to temporary unavailability of the AI service. For a comprehensive analysis, please try again later.
        """

def extract_text_from_pdf(pdf_url):
    # Download the PDF from the URL
    response = requests.get(pdf_url)
    if response.status_code != 200:
        st.error(f"Failed to download the PDF from URL: {pdf_url}")
        return None
    
    # Save the PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name
    
    # Use PyPDFLoader to load the PDF
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()  # Assuming this returns a structured document or text
    
    os.unlink(tmp_file_path)  # Clean up the temporary file

    # Assuming docs is a list of document objects, extract the text
    if docs and isinstance(docs, list):
        text = "\n".join([doc.page_content for doc in docs])  # Adjust based on structure
        return text
    return None


def main():
    st.set_page_config(page_title="AI Medical Report Analyzer", layout="wide")
    st.title("🩺 AI-driven Medical Report Analyzer")
    query_params = st.query_params
    # print(f"Query Parameters: {query_params}")  # Debugging line
# Check if 'pdf_link' exists in the query parameters
    if 'pdf_link' in query_params:
        pdf_url = unquote(query_params['pdf_link'])
        print(f"PDF URL: {pdf_url}")  # Debugging line
        if st.button("🔍 Analyze PDF Report"):
            with st.spinner("Analyzing the medical report..."):
                # Extract text from the PDF at the given URL
                pdf_text = extract_text_from_pdf(pdf_url)
                
                if pdf_text:                    
                    # Perform the analysis on the extracted text
                    analysis = analyze_medical_report(pdf_text, "text")
                    st.subheader("📊 Analysis Results:")
                    st.write(analysis)
                else:
                    st.error("Failed to extract text from the PDF.")
    else:
        st.info("Please provide a PDF link in the URL parameters to analyze a medical report.")
        st.markdown("""
        ### How to use this tool:
        1. Add a PDF link to the URL as a query parameter: `?pdf_link=YOUR_PDF_URL`
        2. The AI will automatically analyze the medical report and provide insights
        """)    
if __name__ == "__main__":
    main()
