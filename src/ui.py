import gradio as gr
from citation import generate_citations
from llm import generate_response
import os

# Handles user query and calls the appropriate retriever and LLM response generator
def handle_query(api_key, query, bm25_retriever, dense_retriever, hybrid_retriever, method):
    if method == "Hybrid":
        results = hybrid_retriever.search(query, top_n=5, use_bm25=True, use_dense=True)
    elif method == "BM25":
        results = bm25_retriever.search(query, top_n=5)
    elif method == "Dense":
        results = dense_retriever.search(query, top_n=5)
    else:
        return "Invalid search method selected!", "", ""

    if not results:
        return "No results found!", "", ""

    # Extract context chunks for response generation
    context_chunks = [result[0] for result in results]
    response = generate_response(query, context_chunks, api_key)

    # Generate a detailed view of retrieved documents
    retrieved_docs_with_context = "\n\n".join(
        [f"Document: {chunk.get('book', chunk.get('source', 'Unknown'))}\nContext: {chunk['text'][:200]}..."
         for chunk, _ in results]
    )

    # Generate citations and sources
    response_with_citations, retrieved_docs, sources = generate_citations(
        response, results, api_key, use_llm=True
    )


    return response_with_citations, retrieved_docs_with_context, sources

# Creates a Gradio UI for the application
def create_ui(bm25_retriever, dense_retriever, hybrid_retriever):
    with gr.Blocks(css=".my-button {background-color: green; color: white; width: 100%;}") as demo:
        gr.Markdown("""
            ## RAG Search with Citations for Harry Potter⚡️
            This project leverages Retrieval-Augmented Generation (RAG) to answer user queries based on the magical world of Harry Potter. 
            Feel free to explore the Hogwarts universe by asking any question about its characters, spells, and magical events!

            The system uses:
            - 7 books from the Harry Potter series
            - 3 Harry Potter movies
            """)

        # Input section for API key and user query
        with gr.Row():
            api_key = gr.Textbox(
                label="API Key", 
                type="password", 
                placeholder="Enter your API key", 
                scale=3
            )
            query = gr.Textbox(
                label="Your Query", 
                placeholder="Example: Who is Harry Potter?", 
                scale=5
            )

        # Search method selection
        with gr.Row():
            method = gr.Radio(
                choices=["BM25", "Dense", "Hybrid"],
                label="Search Method",
                value="BM25",
                interactive=True,
                scale=2
            )

        # Search button
        with gr.Row():
            submit_button = gr.Button(
                "Search 🪄",
                elem_classes="my-button",
                scale=1
            )

        # Outputs: response, retrieved documents, and sources
        with gr.Row():
            response_box = gr.Textbox(
                label="Response⚡️",
                interactive=False,
                lines=5,
                max_lines=5,
                scale=6
            )

        with gr.Row():
            retrieved_docs_box = gr.Textbox(
                label="Retrieved Documents",
                interactive=False,
                lines=7,
                max_lines=7,
                scale=5
            )
            sources_box = gr.Textbox(
                label="Sources",
                interactive=False,
                lines=7,
                max_lines=7,
                scale=5
            )

        # Link the button to the query handling function
        submit_button.click(
            fn=handle_query,
            inputs=[
                api_key,
                query,
                gr.State(bm25_retriever),
                gr.State(dense_retriever),
                gr.State(hybrid_retriever),
                method
            ],
            outputs=[response_box, retrieved_docs_box, sources_box]
        )

    return demo