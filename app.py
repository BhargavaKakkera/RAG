
rom __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import streamlit as st

from chains.model_factory import get_chat_model
from chains.qa_chain import create_conversational_rag_chain
from chains.summarization_chain import generate_summary
from config import settings
from embeddings.factory import get_embedding_model
from loaders.pdf_loader import load_uploaded_pdfs
from loaders.text_splitter import chunk_documents
from memory.session_history import SessionHistoryStore
from prompts.templates import MODE_INSTRUCTIONS
from retrievers.threshold_retriever import ThresholdRetriever
from utils.sources import source_rows
from vectorstore.chroma_store import build_chroma_store


st.set_page_config(
    page_title="Conversational PDF RAG Assistant",
    page_icon="PDF",
    layout="wide",
)


def init_session_state() -> None:
    defaults = {
        "session_id": uuid4().hex,
        "messages": [],
        "documents": [],
        "chunks": [],
        "vectorstore": None,
        "llm": None,
        "history_store": None,
        "last_sources": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_sources = []
    if st.session_state.history_store is not None:
        st.session_state.history_store.clear(st.session_state.session_id)


def build_effective_settings(
    llm_provider: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_k: int,
    similarity_threshold: float,
    api_key_override: str | None,
):
    return replace(
        settings,
        google_api_key=api_key_override or settings.google_api_key,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        default_chunk_size=chunk_size,
        default_chunk_overlap=chunk_overlap,
        default_retrieval_k=retrieval_k,
        default_similarity_threshold=similarity_threshold,
    )


init_session_state()

st.title("Conversational PDF RAG Assistant")
st.caption("Upload one or more PDFs, index them with ChromaDB, and chat with grounded answers.")

with st.sidebar:
    st.header("Settings")

    llm_provider = st.selectbox(
        "LLM provider",
        ["gemini", "ollama"],
        index=0 if settings.llm_provider == "gemini" else 1,
    )
    embedding_provider = st.selectbox(
        "Embedding provider",
        ["gemini", "huggingface", "ollama"],
        index=["gemini", "huggingface", "ollama"].index(settings.embedding_provider)
        if settings.embedding_provider in ["gemini", "huggingface", "ollama"]
        else 0,
    )

    needs_google_key = llm_provider == "gemini" or embedding_provider == "gemini"
    api_key_override = None
    if needs_google_key and not settings.google_api_key:
        api_key_override = st.text_input(
            "Gemini API key",
            type="password",
            help="Used only for this Streamlit session. Prefer .env for local development.",
        )

    st.divider()
    mode = st.selectbox("Answer mode", list(MODE_INSTRUCTIONS.keys()))
    chunk_size = st.slider("Chunk size", 300, 3000, settings.default_chunk_size, 100)
    chunk_overlap = st.slider(
        "Chunk overlap",
        0,
        min(800, chunk_size - 1),
        min(settings.default_chunk_overlap, chunk_size - 1),
        25,
    )
    retrieval_k = st.slider("Retrieved chunks", 1, 12, settings.default_retrieval_k, 1)
    similarity_threshold = st.slider(
        "Similarity threshold",
        0.0,
        1.0,
        settings.default_similarity_threshold,
        0.05,
    )

    if st.button("Clear chat", use_container_width=True):
        reset_chat()
        st.rerun()

effective_settings = build_effective_settings(
    llm_provider,
    embedding_provider,
    chunk_size,
    chunk_overlap,
    retrieval_k,
    similarity_threshold,
    api_key_override,
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

index_col, stats_col = st.columns([1, 2])

with index_col:
    index_clicked = st.button(
        "Index uploaded PDFs",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

with stats_col:
    if st.session_state.chunks:
        st.success(
            f"Indexed {len(st.session_state.documents)} pages into "
            f"{len(st.session_state.chunks)} chunks."
        )
    else:
        st.info("Upload PDFs and click index to begin.")

if index_clicked:
    try:
        with st.spinner("Extracting PDF text..."):
            documents = load_uploaded_pdfs(uploaded_files)

        with st.spinner("Chunking documents..."):
            chunks = chunk_documents(
                documents,
                effective_settings.default_chunk_size,
                effective_settings.default_chunk_overlap,
            )

        with st.spinner("Creating embeddings and storing vectors in ChromaDB..."):
            embedding_model = get_embedding_model(effective_settings)
            collection_name = f"pdf_rag_{uuid4().hex}"
            vectorstore = build_chroma_store(
                chunks,
                embedding_model,
                effective_settings,
                collection_name,
            )

        with st.spinner("Preparing chat model and conversational memory..."):
            llm = get_chat_model(effective_settings)
            history_store = SessionHistoryStore(effective_settings.max_history_messages)

        st.session_state.documents = documents
        st.session_state.chunks = chunks
        st.session_state.vectorstore = vectorstore
        st.session_state.llm = llm
        st.session_state.history_store = history_store
        reset_chat()
        st.success("PDFs indexed successfully. Ask a question below.")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

chat_tab, sources_tab, summary_tab = st.tabs(["Chat", "Sources", "Summaries"])

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about the uploaded PDFs",
        disabled=st.session_state.vectorstore is None,
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                retriever = ThresholdRetriever(
                    vectorstore=st.session_state.vectorstore,
                    k=effective_settings.default_retrieval_k,
                    similarity_threshold=effective_settings.default_similarity_threshold,
                )
                rag_chain = create_conversational_rag_chain(
                    st.session_state.llm,
                    retriever,
                    st.session_state.history_store,
                )

                with st.spinner("Retrieving context and generating answer..."):
                    result = rag_chain.invoke(
                        {"question": question, "mode": mode},
                        config={
                            "configurable": {
                                "session_id": st.session_state.session_id,
                            }
                        },
                    )

                answer = result["answer"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_sources = result.get("source_documents", [])

                if st.session_state.last_sources:
                    with st.expander("Sources used for this answer", expanded=False):
                        st.dataframe(
                            source_rows(st.session_state.last_sources),
                            hide_index=True,
                            use_container_width=True,
                        )
            except Exception as exc:
                error_message = f"Answer generation failed: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

with sources_tab:
    if not st.session_state.last_sources:
        st.info("Ask a question to see retrieved source chunks.")
    else:
        st.subheader("Retrieved Source Chunks")
        st.dataframe(
            source_rows(st.session_state.last_sources),
            hide_index=True,
            use_container_width=True,
        )

        for index, document in enumerate(st.session_state.last_sources, start=1):
            source = document.metadata.get("source", "Unknown source")
            page = document.metadata.get("page", "N/A")
            score = document.metadata.get("similarity_score", "N/A")
            with st.expander(
                f"Chunk {index}: {source}, page {page}, score {score}",
                expanded=index == 1,
            ):
                st.write(document.page_content)

with summary_tab:
    if not st.session_state.documents:
        st.info("Index PDFs first to generate summaries.")
    else:
        summary_type = st.selectbox(
            "Summary type",
            [
                "Complete Document Summary",
                "Chapter/Topic Summaries",
                "Key Points",
            ],
        )
        if st.button("Generate summary", use_container_width=True):
            try:
                with st.spinner("Generating summary..."):
                    summary = generate_summary(
                        st.session_state.llm,
                        st.session_state.documents,
                        summary_type,
                    )
                st.markdown(summary)
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")
