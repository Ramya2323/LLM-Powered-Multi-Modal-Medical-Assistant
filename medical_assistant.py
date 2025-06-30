
import streamlit as st
from PIL import Image
import numpy as np
import faiss
import os
import tempfile
import fitz  # PyMuPDF
import torch

from transformers import CLIPProcessor, CLIPModel

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain_groq import ChatGroq

# ----------- CONFIG -----------
GROQ_API_KEY = "Add your Groq key"
MODEL_NAME = "llama3-8b-8192"

# ----------- SETUP LLM -----------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.3,
)

# ----------- SETUP CLIP MODEL -----------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ----------- STREAMLIT SESSION STATE INIT -----------
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(512)
    st.session_state.stored_images = []

# ----------- AGENTS / TOOLS -----------

# 1. Symptom Checker Tool
symptom_checker = Tool(
    name="Symptom Checker",
    func=lambda q: "Based on your symptoms, this could indicate Dengue, Chikungunya, or a viral fever. Blood test recommended.",
    description="Useful for checking symptoms and suggesting diagnoses."
)

# 2. Report Generator
report_generator = Tool(
    name="Medical Report Generator",
    func=lambda q: f"Patient Summary:\n{q}\n\nLikely Diagnosis: Under Evaluation. Suggested Tests: CBC, CRP, LFT. Treatment: Paracetamol, rest, and fluids.",
    description="Generates medical reports from patient descriptions."
)

# 3. PDF QA Tool
pdf_db = None
pdf_agent_tool = Tool(
    name="Chat with PDF",
    func=lambda q: pdf_db.similarity_search(q, k=2) if pdf_db else "Please upload a PDF first.",
    description="Ask questions based on uploaded medical PDFs."
)

# 4. Image Query Tool (placeholder)
image_query_tool = Tool(
    name="Image Query",
    func=lambda q: f"Analyzing image for: {q}. Please check similar cases and patient history.",
    description="Used to process and compare medical images."
)

# ----------- INITIALIZE AGENT -----------
agent = initialize_agent(
    tools=[symptom_checker, report_generator, pdf_agent_tool, image_query_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

# ----------- HELPER FUNCTIONS -----------
def get_clip_embedding(image_path, text_query):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text_query], images=image, return_tensors="pt", padding=True).to(device)
    clip_model.to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    image_emb = outputs.image_embeds.cpu().numpy()
    text_emb = outputs.text_embeds.cpu().numpy()
    return image_emb, text_emb

def add_image_to_index(image_emb, image_path):
    st.session_state.faiss_index.add(image_emb)
    st.session_state.stored_images.append((image_emb, image_path))
    st.success(f"Image added. Total in FAISS: {st.session_state.faiss_index.ntotal}")

def search_similar_images(image_emb):
    D, I = st.session_state.faiss_index.search(image_emb, k=3)
    return I

def handle_pdf_upload(uploaded_pdf):
    global pdf_db
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings()
    pdf_db = FAISS.from_documents(pages, embeddings)
    return "PDF uploaded and processed."

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="Medical GenAI Assistant", layout="centered")
st.title("🧠 Medical GenAI Assistant (Powered by Groq)")

menu = st.sidebar.radio("Choose a function", (
    "Chat with Agent",
    "Symptom Checker",
    "Upload & Chat with PDF",
    "Upload Medical Image",
    "Query with Image",
    "Generate Medical Report",
    "Debug"
))

if menu == "Chat with Agent":
    q = st.text_input("Ask any medical question")
    if st.button("Run Agent") and q:
        response = agent.run(q)
        st.write("**Response:**", response)

elif menu == "Symptom Checker":
    symptoms = st.text_area("Describe your symptoms")
    if st.button("Check Symptoms") and symptoms:
        response = symptom_checker.func(symptoms)
        st.write("**Assessment:**", response)

elif menu == "Upload & Chat with PDF":
    uploaded_pdf = st.file_uploader("Upload a medical PDF", type="pdf")
    if uploaded_pdf and st.button("Process PDF"):
        st.success(handle_pdf_upload(uploaded_pdf))

    pdf_query = st.text_input("Ask a question about the uploaded PDF")
    if pdf_query and pdf_db:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=pdf_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=False
        )
        answer = qa_chain.run(pdf_query)
        st.markdown(f"**Answer:** {answer}")

elif menu == "Upload Medical Image":
    uploaded_image = st.file_uploader("Upload medical image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_image.read())
            tmp_path = tmp.name
        image_emb, _ = get_clip_embedding(tmp_path, "Medical image")
        st.write(f"Embedding shape: {image_emb.shape}")
        add_image_to_index(image_emb, tmp_path)

    if st.checkbox("Show Stored Images"):
        st.write(f"Total images: {len(st.session_state.stored_images)}")
        for idx, (_, path) in enumerate(st.session_state.stored_images):
            st.image(path, caption=f"Image {idx}", width=200)

elif menu == "Query with Image":
    uploaded_image = st.file_uploader("Upload image for query", type=["jpg", "png", "jpeg"])
    text_context = st.text_input("Describe the context/symptoms")

    if uploaded_image and text_context and st.button("Query"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_image.read())
            tmp_path = tmp.name

        image_emb, text_emb = get_clip_embedding(tmp_path, text_context)
        st.write(f"Embedding shape: {image_emb.shape}")
        st.write(f"FAISS Index Size: {st.session_state.faiss_index.ntotal}")

        if st.session_state.faiss_index.ntotal == 0:
            st.error("No images in FAISS index. Please upload some first.")
        else:
            similar = search_similar_images(image_emb)
            st.markdown(f"**Similar Image IDs:** {similar.tolist()}")

            if all(i == -1 for i in similar[0]):
                st.error("No similar images found.")
            else:
                try:
                    query_input = f"User context: {text_context}. Found similar image IDs: {similar.tolist()}"
                    query_result = agent.run(query_input)
                    st.markdown(f"**Agent Response:** {query_result}")
                except Exception as e:
                    st.error(f"Agent failed: {e}")

elif menu == "Generate Medical Report":
    case_description = st.text_area("Describe the patient case")
    if st.button("Generate Report") and case_description:
        report = report_generator.func(case_description)
        st.markdown(f"**Generated Report:**\n\n{report}")

elif menu == "Debug":
    st.subheader("🛠 Debug Info")
    st.write(f"Total images in memory: {len(st.session_state.stored_images)}")
    st.write(f"FAISS index size: {st.session_state.faiss_index.ntotal}")
    
    if st.button("Test FAISS with random vector"):
        dummy_vec = np.random.rand(1, 512).astype("float32")
        D, I = st.session_state.faiss_index.search(dummy_vec, k=3)
        st.write("Random FAISS result:", I.tolist())
