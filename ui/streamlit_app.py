"""Ask questions about the ingested arXiv papers and see the evidence behind them.

Run it from the project root so the pipeline finds .env and chroma_db:

    uv run streamlit run ui/streamlit_app.py
"""

import random
import time
import uuid

import streamlit as st
from citations import build_sources, link_citation_markers
from pipeline import AnswerChunk, AnswerComplete, stream_answer_in_conversation

# Non-streaming implementation import, kept for quick rollback/testing:
# from concurrent.futures import ThreadPoolExecutor, wait

# Missing .env keys, a missing Chroma database and a rejected answer all reach
# the UI as one of these, and all of them are worth showing the reader.
PIPELINE_ERRORS = (FileNotFoundError, RuntimeError, ValueError)

# The route is unknown until the graph decides, so the waiting label stays neutral
# and simply rotates while the answer is being produced.
THINKING_PHRASES = (
    "Thinking", "Pondering", "Noodling", "Percolating", "Musing",
    "Ruminating", "Cogitating", "Puzzling", "Deliberating", "Mulling",
)
PHRASE_SECONDS = 3.0

SUGGESTIONS = {
    ":blue[:material/hub:] Multi-head attention": "What is multi-head attention, and why use several heads?",
    ":green[:material/school:] Training setup": "How were the models trained, and on what hardware?",
    ":orange[:material/query_stats:] Reported results": "What results are reported, and on which benchmarks?",
}

st.set_page_config(page_title="arXiv reading assistant", page_icon=":material/menu_book:")


def start_conversation() -> None:
    """Empty the chat and start a new trace thread for the next questions."""
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def thinking_labels():
    """Endless stream of neutral waiting words, reshuffled so the order feels fresh."""
    while True:
        words = list(THINKING_PHRASES)
        random.shuffle(words)
        yield from words


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
    st.subheader("Answer style")
    selected_mode = st.segmented_control(
        "Answer style",
        options=["standard", "easy"],
        default="standard",
        required=True,
        format_func=str.title,
        label_visibility="collapsed",
        width="stretch",
    )
    answer_mode = "easy" if selected_mode == "easy" else "standard"
    st.caption("Easy mode uses simpler language and helpful analogies while keeping citations.")

    if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
        start_conversation()
        st.rerun()

st.title("arXiv reading assistant")
st.caption("Answers come only from the ingested papers. Every claim links to the passage it came from.")

if "messages" not in st.session_state:
    start_conversation()

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
        labels = thinking_labels()
        try:
            status = st.status(f":shimmer[{next(labels)}…]", type="compact")
            answer_placeholder = st.empty()
            streamed_answer = ""
            result = None

            for event in stream_answer_in_conversation(
                question, st.session_state.thread_id, answer_mode
            ):
                if isinstance(event, AnswerChunk):
                    streamed_answer += event.text
                    answer_placeholder.markdown(streamed_answer + "▌")
                elif isinstance(event, AnswerComplete):
                    result = event.result

            if result is None:
                raise RuntimeError("The answer stream ended without a completed result.")

            elapsed = time.perf_counter() - started
            if result.answer_type == "rag":
                status.update(
                    label=f"Read the papers in {elapsed:.0f}s",
                    state="complete",
                )
            else:
                status.update(label=f"Answered in {elapsed:.0f}s", state="complete")
        except PIPELINE_ERRORS as error:
            st.error(str(error), icon=":material/error:")
        else:
            if result.answer_type == "rag":
                citations = result.context.citations
                linked_answer = link_citation_markers(result.answer, citations)
                sources = build_sources(result.answer, citations, result.passages_by_id)
            else:
                linked_answer = result.answer
                sources = []

            # Replace the streamed plain-text citation markers with links once
            # the final context arrives, rather than rendering the answer twice.
            answer_placeholder.markdown(linked_answer)
            render_sources(sources)
            st.session_state.messages.append({"question": question, "answer": linked_answer, "sources": sources})

        # Non-streaming version kept temporarily for quick rollback/testing:
        # with st.status(f":shimmer[{next(labels)}…]", type="compact") as status:
        #     with ThreadPoolExecutor(max_workers=1) as pool:
        #         pending = pool.submit(
        #             answer_in_conversation,
        #             question,
        #             st.session_state.thread_id,
        #             answer_mode,
        #         )
        #         while not wait([pending], timeout=PHRASE_SECONDS).done:
        #             status.update(
        #                 label=f":shimmer[{next(labels)}… "
        #                       f"{time.perf_counter() - started:.0f}s]"
        #             )
        #     result = pending.result()
