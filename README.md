# LLM-Powered-Multi-Modal-Medical-Assistant
This project presents a next-generation Medical GenAI Assistant, designed to combine Large Language
Models (LLMs) with Vision-Language Models (VLMs) for an intelligent, adaptive, and clinically useful
healthcare tool.
Built with a multi-modal architecture, the assistant can process both textual and visual medical data, enabling
physicians and researchers to interact with patient reports, symptoms, and medical images through a unified
interface. At its core, the system integrates:
- Groq's LLaMA3-powered LLM, delivering lightning-fast responses for text understanding, medical
reasoning, and case-based diagnosis.
- CLIP (Contrastive Language-Image Pretraining), enabling deep visual comprehension and semantic
similarity search over uploaded medical images like X-rays, MRIs, and scans.
- FAISS Vector Store, used for embedding and retrieving similar image cases, enhancing clinical
context-awareness.
- LangChain Agents, powering task-specific tools such as symptom checkers, medical report generators,
PDF Q&A bots, and image query agents.

The assistant supports:
- Real-time querying of uploaded medical PDFs, using HuggingFace embeddings and PyMuPDF for dynamic
retrieval-augmented generation (RAG).
- Image-to-text grounding, where medical images are matched semantically to clinical queries and used to
suggest diagnoses or retrieve similar past cases.
- Agent orchestration, where tools dynamically respond to patient descriptions, medical reports, or diagnostic
queries based on symptom data and imaging.
Deployed via Streamlit, the assistant features an intuitive interface that supports clinicians with:
- Symptom triage and disease suggestion
- Automated medical report generation
-  Interactive PDF-based Q&A
- Visual-case matching via FAISS + CLIP
This work bridges the gap between unstructured clinical data and structured diagnostics, offering a scalable
and responsive AI system for next-gen medical applications.
