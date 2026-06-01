import streamlit as st
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8002/api/v1"

st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal RAG System")
st.caption("Search across PDFs, CSVs, images, and code files")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["Query", "Upload Documents", "Document Library", "Evaluation Dashboard"]
)

# ─────────────────────────────────────────
# PAGE 1: QUERY
# ─────────────────────────────────────────
if page == "Query":
    st.header("Ask a Question")

    with st.form("query_form"):
        query = st.text_area("Your question", placeholder="e.g. What are the salaries in the CSV?", height=100)
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Results to retrieve", 1, 10, 5)
        with col2:
            file_type = st.selectbox("Filter by type", ["All", "pdf", "csv", "code", "image"])
        with col3:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    if submitted and query:
        with st.spinner("Searching and generating answer..."):
            payload = {"query": query, "top_k": top_k}
            if file_type != "All":
                payload["file_type_filter"] = file_type

            try:
                response = requests.post(f"{API_URL}/query", json=payload)
                result = response.json()

                if response.status_code == 200:
                    # Answer
                    st.subheader("Answer")
                    st.write(result["answer"])

                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Confidence", f"{result['confidence']:.0%}")
                    col2.metric("Retrieval", f"{result['retrieval_latency_ms']:.0f}ms")
                    col3.metric("Generation", f"{result['generation_latency_ms']:.0f}ms")
                    col4.metric("Citations", len(result["citations"]))

                    # Sources
                    if result["sources"]:
                        st.subheader("Sources")
                        for source in result["sources"]:
                            with st.expander(f"📄 {source['filename']} (confidence: {source['confidence']:.0%})"):
                                st.write(f"**File type:** {source['file_type']}")
                                st.write(f"**Page:** {source['page_number'] or 'N/A'}")
                                st.write(f"**Chunk ID:** {source['chunk_id']}")

                    # Rating
                    st.subheader("Rate this answer")
                    query_id = result.get("query_id")
                    rating = st.radio("", [1, 2, 3, 4, 5], horizontal=True, index=4)
                    if st.button("Submit Rating"):
                        history_response = requests.get(f"{API_URL}/history?limit=1")
                        if history_response.status_code == 200:
                            latest = history_response.json()
                            if latest:
                                qid = latest[0]["id"]
                                requests.post(f"{API_URL}/history/{qid}/rate?rating={rating}")
                                st.success(f"Rated {rating}/5 ⭐")
                else:
                    st.error(f"Error: {result.get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running on port 8002.")

# ─────────────────────────────────────────
# PAGE 2: UPLOAD
# ─────────────────────────────────────────
elif page == "Upload Documents":
    st.header("Upload Documents")
    st.write("Supported formats: PDF, CSV, Python, JavaScript, TypeScript, Java, Go, Rust, Markdown, PNG, JPG")

    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        accept_multiple_files=True,
        type=["pdf", "csv", "py", "js", "ts", "java", "go", "rs", "md", "png", "jpg", "jpeg"]
    )

    if uploaded_files:
        if st.button("📥 Ingest All Files", use_container_width=True):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        response = requests.post(f"{API_URL}/ingest", files=files)
                        result = response.json()

                        if response.status_code == 200:
                            st.success(f"✅ {uploaded_file.name} ingested! ({result['metadata']['total_chunks']} chunks)")
                        else:
                            st.error(f"❌ {uploaded_file.name}: {result.get('detail', 'Error')}")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")

# ─────────────────────────────────────────
# PAGE 3: DOCUMENT LIBRARY
# ─────────────────────────────────────────
elif page == "Document Library":
    st.header("Document Library")

    try:
        response = requests.get(f"{API_URL}/documents")
        documents = response.json()

        if not documents:
            st.info("No documents ingested yet. Go to Upload Documents to add some!")
        else:
            st.write(f"**{len(documents)} documents** ingested")

            for doc in documents:
                icon = {"pdf": "📄", "csv": "📊", "code": "💻", "image": "🖼️"}.get(doc["file_type"], "📁")
                with st.expander(f"{icon} {doc['filename']} — {doc['file_type'].upper()}"):
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**Status:** {doc['status']}")
                    col2.write(f"**Size:** {doc['file_size'] or 0:,} bytes")
                    col3.write(f"**Chunks:** {doc['metadata']['total_chunks'] if doc['metadata'] else 'N/A'}")

                    if doc["page_count"]:
                        st.write(f"**Pages:** {doc['page_count']}")

                    st.write(f"**Ingested:** {doc['created_at'][:10]}")

                    if st.button(f"🗑️ Delete", key=f"del_{doc['id']}"):
                        del_response = requests.delete(f"{API_URL}/documents/{doc['id']}")
                        if del_response.status_code == 200:
                            st.success("Deleted!")
                            st.rerun()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")

# ─────────────────────────────────────────
# PAGE 4: EVALUATION DASHBOARD
# ─────────────────────────────────────────
elif page == "Evaluation Dashboard":
    st.header("Evaluation Dashboard")

    try:
        response = requests.get(f"{API_URL}/evaluate")
        report = response.json()

        # Latency
        st.subheader("⚡ Latency")
        latency = report.get("latency", {})
        if "total_queries" in latency:
            st.write(f"Total queries: **{latency['total_queries']}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Retrieval**")
                ret = latency.get("retrieval", {})
                st.metric("Mean", f"{ret.get('mean_ms', 0):.0f}ms")
                st.metric("Median", f"{ret.get('median_ms', 0):.0f}ms")
                st.metric("Min / Max", f"{ret.get('min_ms', 0):.0f} / {ret.get('max_ms', 0):.0f}ms")
            with col2:
                st.write("**Generation**")
                gen = latency.get("generation", {})
                st.metric("Mean", f"{gen.get('mean_ms', 0):.0f}ms")
                st.metric("Median", f"{gen.get('median_ms', 0):.0f}ms")
                st.metric("Min / Max", f"{gen.get('min_ms', 0):.0f} / {gen.get('max_ms', 0):.0f}ms")

        # Confidence
        st.subheader("🎯 Confidence Distribution")
        conf = report.get("confidence", {})
        if "total_queries" in conf:
            st.metric("Mean Confidence", f"{conf.get('mean_confidence', 0):.0%}")
            dist = conf.get("distribution", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("🟢 High (>80%)", dist.get("high (>0.8)", 0))
            col2.metric("🟡 Medium (50-80%)", dist.get("medium (0.5-0.8)", 0))
            col3.metric("🔴 Low (<50%)", dist.get("low (<0.5)", 0))

        # Query History
        st.subheader("📜 Recent Queries")
        hist_response = requests.get(f"{API_URL}/history?limit=10")
        history = hist_response.json()

        for record in history:
            rating_stars = "⭐" * record["rating"] if record["rating"] else "Not rated"
            with st.expander(f"Q: {record['query'][:60]}... | Confidence: {record['confidence']:.0%} | {rating_stars}"):
                st.write(f"**Answer:** {record['answer']}")
                col1, col2 = st.columns(2)
                col1.write(f"**Retrieval:** {record['retrieval_latency_ms']:.0f}ms")
                col2.write(f"**Generation:** {record['generation_latency_ms']:.0f}ms")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
