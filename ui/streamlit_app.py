"""Ask questions about the ingested arXiv papers and see the evidence behind them.

Run it from the project root so the pipeline finds .env and chroma_db:

    uv run streamlit run ui/streamlit_app.py
"""

import time

import streamlit as st

from arxiv_rag.answering import generate_answer
from arxiv_rag.retrieval import DEFAULT_TOP_K
from citations import build_sources, link_citation_markers
from pipeline import retrieve_evidence

# Missing .env keys, a missing Chroma database and a rejected answer all reach
# the UI as one of these, and all of them are worth showing the reader.
PIPELINE_ERRORS = (FileNotFoundError, RuntimeError, ValueError)

SUGGESTIONS = {
    ":blue[:material/hub:] Multi-head attention": "What is multi-head attention, and why use several heads?",
    ":green[:material/school:] Training setup": "How were the models trained, and on what hardware?",
    ":orange[:material/query_stats:] Reported results": "What results are reported, and on which benchmarks?",
}

st.set_page_config(page_title="arXiv reading assistant", page_icon=":material/menu_book:")


def render_sources(sources: list[dict]) -> None:
    """Show the exact passage behind every citation the answer used."""
    if not sources:
        return

    label = "1 source" if len(sources) == 1 else f"{len(sources)} sources"
    with st.expander(label, icon=":material/menu_book:"):
        for source in sources:
            st.markdown(f"**[{source['number']}]** [{source['label']}]({source['url']})")
            st.caption(source["text"])


with st.sidebar:
    st.subheader("Retrieval")
    top_k = st.slider("Papers searched per question", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    st.caption("Each paper contributes several passages, so the answer usually cites more sources than this.")
    if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.title("arXiv reading assistant")
st.caption("Answers come only from the ingested papers. Every claim links to the passage it came from.")

if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Ask about the papers", submit_mode="disable")

for message in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(message["question"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        render_sources(message["sources"])

if not question and not st.session_state.messages:
    suggestion = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed")
    if suggestion:
        question = SUGGESTIONS[suggestion]

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            with st.status(":shimmer[Reading the papers]", type="compact") as status:
                with st.status("Searching for passages", type="step"):
                    evidence = retrieve_evidence(question, top_k)
                    st.write(f"Found {len(evidence.context.citations)} passages.")

                with st.status("Writing a grounded answer", type="step"):
                    answer = generate_answer(question, evidence.context)

                elapsed = time.perf_counter() - started
                status.update(label=f"Read the papers in {elapsed:.0f}s", state="complete")
        except PIPELINE_ERRORS as error:
            st.error(str(error), icon=":material/error:")
        else:
            citations = evidence.context.citations
            linked_answer = link_citation_markers(answer, citations)
            sources = build_sources(answer, citations, evidence.passages_by_id)

            st.markdown(linked_answer)
            render_sources(sources)
            st.session_state.messages.append({"question": question, "answer": linked_answer, "sources": sources})
