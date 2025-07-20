import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from transformers import pipeline
from typing import List

st.set_page_config(page_title="AI Claims Automation System", layout="wide")
st.title("🤖 AI-Powered Claims Automation")
st.markdown("Streamline insurance workflows with smart document analysis ✨")

# Load a small, efficient zero-shot classifier
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

# Define some example claim types and priorities
CLAIM_TYPES = ["Accident", "Health", "Property Damage", "Theft", "Fire"]
PRIORITIES = ["Low", "Medium", "High"]

# Extract text from uploaded PDF
def extract_text_from_pdf(pdf_file) -> str:
    pdf = PdfReader(pdf_file)
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text

# Classify claim type and priority
def classify_claim(text: str):
    type_result = classifier(text, CLAIM_TYPES)
    priority_result = classifier(text, PRIORITIES)
    return type_result["labels"][0], priority_result["labels"][0]

# Streamlit UI
uploaded_file = st.file_uploader("📄 Upload a claim document (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and analyzing the claim..."):
        # Extract text
        text = extract_text_from_pdf(uploaded_file)

        # Run classification
        claim_type, priority = classify_claim(text)

        # Display results
        st.success("✅ Document processed!")
        st.subheader("Claim Summary")

        st.markdown(f"**Claim Type:** `{claim_type}`")
        st.markdown(f"**Priority Level:** `{priority}`")

        st.subheader("📑 Extracted Text")
        st.text_area("Document Content", text, height=300)

        st.subheader("📌 Routing Decision")
        if priority == "High":
            st.warning("🚨 Route to Senior Claims Officer Immediately")
        elif priority == "Medium":
            st.info("🔄 Route to Standard Review Queue")
        else:
            st.success("✅ Can be auto-approved or batched")

else:
    st.info("👈 Upload a PDF claim report to get started!")
