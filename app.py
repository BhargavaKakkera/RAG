
from __future__ import annotations

import logging
import re
from dataclasses import replace

import streamlit as st
from uuid import uuid4

from chains.model_factory import get_chat_model

from chains.qa_chain import create_conversational_rag_chain
from chains.summarization_chain import generate_summary
from config import settings
from embeddings.factory import get_embedding_model
from loaders.pdf_loader import load_uploaded_pdfs
from loaders.text_splitter import chunk_documents
from memory.session_history import SessionHistoryStore
from prompts.templates import DOCUMENT_METADATA_PROMPT
from retrievers.threshold_retriever import ThresholdRetriever
from utils.sources import source_rows
from utils.timing import log_timing
from vectorstore.chroma_store import (
    compute_upload_fingerprint,
    get_or_build_chroma_store,
    load_document_metadata,
    save_document_metadata,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.app")


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
        "llm_fingerprint": None,
        "history_store": None,
        "last_sources": [],
        "indexed_fingerprint": None,
        "document_metadata": None,
        "summary_cache": {},
        "last_index_banner_fingerprint": None,
        "show_index_banner": False,
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
    llm_model: str,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_k: int,
    similarity_threshold: float,
    prompt_context_budget: int,
):
    return replace(
        settings,
        google_api_key=settings.google_api_key,
        groq_api_key=settings.groq_api_key,

        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        gemini_model=llm_model if llm_provider == "gemini" else settings.gemini_model,
        groq_model=llm_model if llm_provider == "groq" else settings.groq_model,
        ollama_model=llm_model if llm_provider == "ollama" else settings.ollama_model,
        default_chunk_size=chunk_size,
        default_chunk_overlap=chunk_overlap,
        default_retrieval_k=retrieval_k,
        default_similarity_threshold=similarity_threshold,
        default_prompt_context_budget=prompt_context_budget,
    )


def _detect_intent(question: str) -> str:
    q = question.strip().lower()

    q_norm = re.sub(r"\s+", " ", q)

    overview_patterns = [
        r"\bwhat is in (the|this|my) (pdf|document)\b",
        r"\bwhat does (the|this|my) (pdf|document) contain\b",
        r"\bwhat is (the|this|my) (pdf|document) about\b",
        r"\b(explain|describe) (the|this|my) (pdf|document)\b",
        r"\bgive (me )?an overview\b",
        r"\bdocument overview\b",
        r"\bwho is the author\b",
    ]
    if any(re.search(pattern, q_norm) for pattern in overview_patterns):
        return "document_overview"

    list_topics_patterns = [
        r"\blist (all )?(the )?(main )?topics\b",
        r"\blist (the )?themes\b",
        r"\bwhat topics\b",
        r"\bkey topics\b",
        r"\bmain topics\b",
    ]
    if any(re.search(pattern, q_norm) for pattern in list_topics_patterns):
        return "list_topics"

    toc_patterns = [
        r"\btable of contents\b",
        r"\btable of content\b",
        r"\bchapter list\b",
        r"\blist (the )?chapters\b",
        r"\bwhat chapters\b",
        r"\bcontents of (the|this) (pdf|document)\b",
    ]
    if any(re.search(pattern, q_norm) for pattern in toc_patterns):
        return "table_of_contents"

    summary_patterns = [
        r"\bcomplete document summary\b",
        r"\bchapter summaries\b",
        r"\btopic summaries\b",
        r"\bkey points\b",
        r"\bsummarize (the|this) (pdf|document|book)\b",
        r"\bgive (me )?(a )?summary\b",
        r"\bexecutive summary\b",
    ]
    if any(re.search(pattern, q_norm) for pattern in summary_patterns):
        return "summary_request"

    if re.search(r"\b(thank you|thanks|thx)\b", q_norm):
        return "thanks"

    if re.search(r"\b(bye|goodbye|see you|farewell)\b", q_norm):
        return "goodbye"

    greeting_patterns = [
        r"^(hi|hello|hey)([!?.\s]|$)",
        r"^good morning([!?.\s]|$)",
        r"^good afternoon([!?.\s]|$)",
        r"^good evening([!?.\s]|$)",
    ]
    if any(re.search(pattern, q_norm) for pattern in greeting_patterns):
        return "greeting"

    return "question_answering"


def _select_summary_type_from_question(question: str) -> str:
    q = question.strip().lower()
    if re.search(r"\bkey points\b", q):
        return "Key Points"
    if re.search(r"\b(chapter|topic|themes)\b", q):
        return "Chapter/Topic Summaries"
    return "Complete Document Summary"


def _format_document_overview(metadata: dict, sources: list[str]) -> str:
    parts: list[str] = []
    for source in sources:
        doc_meta = metadata.get(source, {})
        title = doc_meta.get("title", source)
        summary = doc_meta.get("summary")
        page_count = doc_meta.get("page_count", "N/A")
        if summary:
            parts.append(f"**{title}** ({page_count} pages)\n{summary}")
        else:
            parts.append(f"**{title}** ({page_count} pages)\n(No summary available.)")
    return "\n\n".join(parts).strip()


def _format_topics(metadata: dict, sources: list[str]) -> str:
    parts: list[str] = []
    for source in sources:
        doc_meta = metadata.get(source, {})
        title = doc_meta.get("title", source)
        topics = doc_meta.get("topics") or []
        keywords = doc_meta.get("keywords") or []
        topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        section = f"**{title}**\nTopics: {topics_str or 'N/A'}"
        if keywords_str:
            section += f"\nKeywords: {keywords_str}"
        parts.append(section)
    return "\n\n".join(parts).strip()


def _format_toc(metadata: dict, sources: list[str]) -> str:
    parts: list[str] = []
    for source in sources:
        doc_meta = metadata.get(source, {})
        title = doc_meta.get("title", source)
        toc = doc_meta.get("table_of_contents")
        if toc:
            parts.append(f"**{title}**\n{toc}")
        else:
            parts.append(f"**{title}**\n(No table of contents available.)")
    return "\n\n".join(parts).strip()


def _generate_document_metadata(
    llm,
    documents,
    selected_sources: list[str] | None = None,
) -> dict:

    grouped: dict[str, list] = {}

    for document in documents:
        source = document.metadata.get("source", "Unknown source")
        grouped.setdefault(source, []).append(document)

    if selected_sources is None:
        selected_sources = list(grouped.keys())

    max_chars_per_doc = 25000
    metadata: dict[str, dict] = {}

    with log_timing("metadata_generation"):
        for source in selected_sources:
            pages = grouped.get(source, [])
            page_count = len({page.metadata.get("page") for page in pages}) if pages else 0
            ordered_pages = sorted(pages, key=lambda item: item.metadata.get("page", 0))

            parts: list[str] = []
            used = 0
            for document in ordered_pages:
                part = f"[Page {document.metadata.get('page', 'N/A')}]\n{document.page_content}"
                if parts and used + len(part) > max_chars_per_doc:
                    break
                if used + len(part) > max_chars_per_doc:
                    remaining = max_chars_per_doc - used
                    part = part[: max(0, remaining)]
                parts.append(part)
                used += len(part)
                if used >= max_chars_per_doc:
                    break

            context = "\n\n".join(parts)
            chain = DOCUMENT_METADATA_PROMPT | llm
            from utils.llm_invoke import invoke_llm

            # Metadata retry/fallback is scoped to this request.
            provider = effective_settings.llm_provider
            original_model = (
                getattr(llm, "model", None)
                or getattr(llm, "model_name", None)
                or (effective_settings.groq_model if provider == "groq" else "unknown")
            )

            def _set_model(m: str):
                # Some LCEL/LLM wrappers may expose frozen properties like `mode`.
                # Best-effort: only set `model` if it is actually mutable.
                try:
                    setattr(llm, "model", m)
                except Exception:
                    pass

            raw = invoke_llm(
                provider=provider,
                original_model=original_model,
                chain_name="Metadata",
                fallback_model="meta-llama/llama-4-scout-17b-16e-instruct",
                model_setter=_set_model,
                chain_callable=lambda: chain.invoke(
                    {"context": context, "title": source, "page_count": page_count}
                ),
                inputs={"context": context, "title": source, "page_count": page_count},
            )

            raw_text = getattr(raw, "content", raw)


            try:
                import json

                parsed = json.loads(raw_text)
            except Exception:
                parsed = {
                    "title": source,
                    "summary": str(raw_text),
                    "topics": [],
                    "keywords": [],
                    "table_of_contents": None,
                    "page_count": page_count,
                }

            parsed.setdefault("title", source)
            parsed.setdefault("page_count", page_count)
            metadata[source] = parsed

    return metadata


def _ensure_metadata_and_summary_cache(
    llm,
    documents,
    uploaded_files,
    effective_settings,
    force_regenerate: bool = False,
) -> None:


    fingerprint = compute_upload_fingerprint(uploaded_files)
    if (
        not force_regenerate
        and st.session_state.indexed_fingerprint == fingerprint
        and st.session_state.document_metadata is not None
    ):
        return

    st.session_state.indexed_fingerprint = fingerprint
    cached_metadata = load_document_metadata(effective_settings, fingerprint)
    if cached_metadata and not force_regenerate:
        st.session_state.document_metadata = cached_metadata
    else:
        with st.spinner("Generating document metadata (title, summary, topics, keywords, TOC)..."):
            st.session_state.document_metadata = _generate_document_metadata(llm, documents)
            save_document_metadata(
                effective_settings,
                fingerprint,
                st.session_state.document_metadata,
            )

    st.session_state.summary_cache = {}


init_session_state()

st.title("Conversational PDF RAG Assistant")
st.caption("Upload one or more PDFs, index them with ChromaDB, and chat with grounded answers.")

with st.sidebar:
    st.header("Settings")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["groq", "ollama", "gemini"],
        index=["groq", "ollama", "gemini"].index(settings.llm_provider)
        if settings.llm_provider in ["groq", "ollama", "gemini"]
        else 0,
    )

    embedding_provider = st.selectbox(
        "Embedding Provider",
        ["huggingface", "ollama", "gemini"],
        index=["huggingface", "ollama", "gemini"].index(settings.embedding_provider)
        if settings.embedding_provider in ["huggingface", "ollama", "gemini"]
        else 0,
    )


    if llm_provider == "groq":
        llm_model = st.selectbox(
            "Model Selection",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "openai/gpt-oss-120b",
                "groq/compound",
                "meta-llama/llama-4-scout-17b-16e-instruct",
            ],
        )
    elif llm_provider == "gemini":
        llm_model = st.selectbox(
            "Model Selection",
            [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-pro",
            ],
            index=0,
        )
    else:
        llm_model = st.text_input("Llama Model (via Ollama)", value=settings.ollama_model)





    st.divider()
    chunk_size = st.slider("Chunk Size", 100, 3000, settings.default_chunk_size, 100)
    chunk_overlap = st.slider(
        "Chunk Overlap",
        0,
        min(1000, max(chunk_size - 1, 0)),
        min(settings.default_chunk_overlap, max(chunk_size - 1, 0)),
        25,
    )
    retrieval_k = st.slider("Retrieval K", 1, 30, settings.default_retrieval_k, 1)
    similarity_threshold = st.slider(
        "Similarity Threshold",
        0.0,
        1.0,
        settings.default_similarity_threshold,
        0.05,
    )
    default_prompt_budget = (
        settings.default_prompt_context_budget_ollama
        if llm_provider == "ollama"
        else settings.default_prompt_context_budget
    )
    prompt_context_budget = st.slider(
        "Prompt Context Budget",
        1000,
        50000,
        default_prompt_budget,
        500,
        help="Maximum characters of retrieved context sent to the LLM.",
    )

    if st.button("Clear chat", use_container_width=True):
        reset_chat()
        st.rerun()

effective_settings = build_effective_settings(
    llm_provider,
    embedding_provider,
    llm_model,
    chunk_size,
    chunk_overlap,
    retrieval_k,
    similarity_threshold,
    prompt_context_budget,
)


def current_llm_fingerprint(settings) -> str:

    return (
        f"{settings.llm_provider}:"
        f"{settings.gemini_model}:"
        f"{settings.groq_model}:"
        f"{settings.ollama_model}"
    )


uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# ---- Auto‑reset caches when the uploaded file set changes ----
if uploaded_files:
    current_fingerprint = compute_upload_fingerprint(uploaded_files)
else:
    current_fingerprint = None

prev_fp = st.session_state.get("uploaded_fingerprint")
if current_fingerprint != prev_fp:
    # Store the new fingerprint
    st.session_state.uploaded_fingerprint = current_fingerprint

    # Clear caches that depend on the uploaded PDFs
    st.session_state.documents = []
    st.session_state.vectorstore = None
    st.session_state.document_metadata = None
    st.session_state.summary_cache = {}
    st.session_state.last_sources = []
    st.session_state.indexed_fingerprint = None
    # Optional: clear the chat history to avoid confusion
    st.session_state.messages = []

index_col, stats_col = st.columns([1, 2])

with index_col:
    index_clicked = st.button(
        "Index uploaded PDFs",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

with stats_col:
    

    current_fingerprint = (
        compute_upload_fingerprint(uploaded_files)
        if uploaded_files
        else None
    )

    if (
        st.session_state.show_index_banner
        and st.session_state.last_index_banner_fingerprint == current_fingerprint
    ):
        st.success(
            f"Indexed {len(st.session_state.documents)} pages into "
            f"{len(st.session_state.chunks)} chunks."
        )
    elif not st.session_state.chunks:
        st.info("Upload PDFs and click index to begin.")


if index_clicked:
    st.session_state.show_index_banner = False
    st.session_state.last_index_banner_fingerprint = None

    with st.status("Indexing uploaded PDFs...", expanded=True) as status:
        try:
            documents = load_uploaded_pdfs(uploaded_files)

            with log_timing("chunking"):
                chunks = chunk_documents(
                    documents,
                    effective_settings.default_chunk_size,
                    effective_settings.default_chunk_overlap,
                )

            fingerprint = compute_upload_fingerprint(
                uploaded_files,
                chunk_size=effective_settings.default_chunk_size,
                chunk_overlap=effective_settings.default_chunk_overlap,
            )
            embedding_model = get_embedding_model(effective_settings)

            with log_timing("embedding_generation"):
                vectorstore, reused_collection = get_or_build_chroma_store(
                    documents,
                    chunks,
                    embedding_model,
                    effective_settings,
                    fingerprint,
                )

            if reused_collection:
                logger.info(
                    "Reused existing Chroma collection for fingerprint %s", fingerprint
                )
                st.session_state.chunks = chunks
            else:
                st.session_state.chunks = chunks

            with log_timing(
                "model_loading", provider=effective_settings.llm_provider
            ):
                llm = get_chat_model(effective_settings)

            st.session_state.llm_fingerprint = current_llm_fingerprint(
                effective_settings
            )

            with log_timing("history_setup"):
                history_store = SessionHistoryStore(
                    effective_settings.max_history_messages
                )

            st.session_state.documents = documents
            st.session_state.vectorstore = vectorstore
            st.session_state.llm = llm
            st.session_state.history_store = history_store

            # Lazy metadata generation avoids latency on initial PDF indexing.
            reset_chat()

            st.session_state.show_index_banner = True
            st.session_state.last_index_banner_fingerprint = fingerprint

            reuse_note = (
                " Reused existing embeddings." if reused_collection else ""
            )
            pages_count = len(documents)
            st.success(
                f"PDFs indexed successfully.{reuse_note} "
                f"(Extracted {pages_count} pages → {len(chunks)} chunks) "
                f"Ask a question below."
            )

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

        intent = _detect_intent(question)
        logger.info("Detected intent: %s", intent)

        predefined_responses = {
            "greeting": "Hello! Upload PDFs, click **Index uploaded PDFs**, and ask a question about the document content.",
            "goodbye": "Goodbye! If you upload more PDFs or ask another question, I’ll be here.",
            "thanks": "You’re welcome! Let me know if you’d like an overview, topics, table of contents, or a grounded answer.",
        }

        with st.chat_message("assistant"):
            if intent in predefined_responses:
                answer = predefined_responses[intent]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_sources = []
            else:
                try:
                    # Ensure document metadata exists before formatting.
                    # Note: _ensure_metadata_and_summary_cache already shows its own
                    # st.spinner internally — no outer spinner needed here.
                    if st.session_state.document_metadata is None:
                        _ensure_metadata_and_summary_cache(
                            st.session_state.llm,
                            st.session_state.documents,
                            uploaded_files,
                            effective_settings,
                            force_regenerate=False,
                        )

                    sources = list(st.session_state.document_metadata.keys())


                    if intent == "document_overview":
                        answer = _format_document_overview(
                            st.session_state.document_metadata,
                            sources,
                        )
                    elif intent == "list_topics":
                        answer = _format_topics(st.session_state.document_metadata, sources)
                    elif intent == "table_of_contents":
                        answer = _format_toc(st.session_state.document_metadata, sources)

                    elif intent == "summary_request":
                        if st.session_state.document_metadata is None:
                            st.info("Index PDFs first (or generate metadata) to create summaries.")
                            with st.spinner(
                                "Generating document metadata (title/summary/topics/TOC)..."
                            ):
                                _ensure_metadata_and_summary_cache(
                                    st.session_state.llm,
                                    st.session_state.documents,
                                    uploaded_files,
                                    effective_settings,
                                    force_regenerate=False,
                                )

                        summary_type = _select_summary_type_from_question(question)
                        cache_key = (st.session_state.indexed_fingerprint, summary_type)


                        if cache_key in st.session_state.summary_cache:
                            answer = st.session_state.summary_cache[cache_key]
                        else:
                            with st.spinner("Generating cached summary..."):
                                answer = generate_summary(
                                    st.session_state.llm,
                                    st.session_state.documents,
                                    summary_type,
                                    llm_provider=effective_settings.llm_provider,
                                )
                            st.session_state.summary_cache[cache_key] = answer
                    else:
                        retriever = ThresholdRetriever(
                            vectorstore=st.session_state.vectorstore,
                            k=effective_settings.default_retrieval_k,
                            similarity_threshold=effective_settings.default_similarity_threshold,
                        )
                        rag_chain = create_conversational_rag_chain(
                            st.session_state.llm,
                            retriever,
                            st.session_state.history_store,
                            max_context_chars=effective_settings.default_prompt_context_budget,
                            max_retrieval_k=effective_settings.default_retrieval_k,
                            max_rewrite_history_messages=effective_settings.max_rewrite_history_messages,
                            llm_provider=effective_settings.llm_provider,
                            llm_model=(
                                effective_settings.gemini_model
                                if effective_settings.llm_provider == "gemini"
                                else (
                                    effective_settings.groq_model
                                    if effective_settings.llm_provider == "groq"
                                    else effective_settings.ollama_model
                                )
                            ),
                        )

                        with st.spinner("Retrieving context and generating answer..."):
                            result = rag_chain.invoke(
                                {"question": question},
                                config={
                                    "configurable": {
                                        "session_id": st.session_state.session_id,
                                    }
                                },
                            )

                        answer = result["answer"]
                        st.session_state.last_sources = result.get("source_documents", [])


                        if st.session_state.last_sources:
                            with st.expander("Sources used for this answer", expanded=False):
                                st.dataframe(
                                    source_rows(st.session_state.last_sources),
                                    hide_index=True,
                                    use_container_width=True,
                                )

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    if intent in {
                        "document_overview",
                        "list_topics",
                        "table_of_contents",
                        "summary_request",
                    }:
                        st.session_state.last_sources = []
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
                selected_fingerprint = current_llm_fingerprint(effective_settings)
                if (
                    st.session_state.llm is None
                    or st.session_state.llm_fingerprint != selected_fingerprint
                ):
                    with st.spinner("Loading selected LLM..."):
                        st.session_state.llm = get_chat_model(effective_settings)
                        st.session_state.llm_fingerprint = selected_fingerprint

                cache_key = (st.session_state.indexed_fingerprint, summary_type)

                if cache_key in st.session_state.summary_cache:
                    st.markdown(st.session_state.summary_cache[cache_key])
                else:
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(
                            st.session_state.llm,
                            st.session_state.documents,
                            summary_type,
                            llm_provider=effective_settings.llm_provider,
                        )
                    st.session_state.summary_cache[cache_key] = summary
                    st.markdown(summary)
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")

