import os
from chunking import process_books, save_chunks, process_csv, merge_chunks
from llm import generate_response
from retriever import BM25Retriever, DenseRetriever, HybridRetriever
from citation import generate_citations
from ui import create_ui

# Prepares data by processing books and metadata, then merging them
def initialize_data():
    books_dir = 'datasets/books'  # Directory containing book files
    metadata_dir = 'datasets/metadata'  # Directory containing metadata files
    output_dir = 'processed_data'  # Directory for processed data

    # Check if directories exist
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory '{books_dir}' not found!")
    if not os.path.exists(metadata_dir):
        raise FileNotFoundError(f"Metadata directory '{metadata_dir}' not found!")

    # Create the directory for processed data if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("Processing books...")
        book_chunks = process_books(books_dir)
        save_chunks(book_chunks, os.path.join(output_dir, 'chunks_books.json'))

        print("Processing metadata...")
        csv_chunks = process_csv(metadata_dir)
        save_chunks(csv_chunks, os.path.join(output_dir, 'chunks_metadata.json'))

        # Merge book and metadata chunks into a single file
        merged_file = os.path.join(output_dir, 'merged_chunks.json')
        merge_chunks(
            [os.path.join(output_dir, 'chunks_books.json'),
             os.path.join(output_dir, 'chunks_metadata.json')],
            output_file=merged_file
        )

        print("Chunking completed and merged into one file!")
        return merged_file
    except Exception as e:
        print(f"An error occurred during data initialization: {e}")
        raise

# Initializes retrievers: BM25, Dense, and Hybrid
def initialize_retrievers(merged_file, embedding_path='processed_data/embeddings.pkl'):
    print("Initializing retrievers...")
    bm25_retriever = BM25Retriever(merged_file)  # Initialize BM25 retriever
    dense_retriever = DenseRetriever(
        'sentence-transformers/all-MiniLM-L6-v2', merged_file, embedding_path
    )  # Initialize Dense retriever with pre-trained model
    hybrid_retriever = HybridRetriever(bm25_retriever, dense_retriever)  # Combine retrievers
    return bm25_retriever, dense_retriever, hybrid_retriever

# Handles queries by invoking the appropriate retriever and generating citations
def handle_query(api_key, query, bm25_retriever, dense_retriever, hybrid_retriever, method):
    if not query.strip():
        return "Query cannot be empty!", [], [], []

    if method == "Hybrid":
        results = hybrid_retriever.search(query, top_n=5, use_bm25=True, use_dense=True)
    elif method == "BM25":
        results = bm25_retriever.search(query, top_n=5)
    elif method == "Dense":
        results = dense_retriever.search(query, top_n=5)
    else:
        return "Invalid search method selected!", [], [], []

    if not results or not all(isinstance(result, tuple) and len(result) > 0 for result in results):
        return "No results found or results are invalid!", [], [], []

    try:
        # Extract context chunks for response generation
        context_chunks = [result[0] for result in results]
        response = generate_response(query, context_chunks, api_key=api_key)

        # Generate citations for the response
        response_with_citations, retrieved_docs, citations = generate_citations(response, results)

        return response_with_citations, retrieved_docs, citations
    except Exception as e:
        return f"An error occurred during query handling: {e}", [], [], []

# Main function to initialize data, retrievers, and launch the UI
def main():
    try:
        # Initialize the data and retrievers
        merged_file = initialize_data()
        bm25_retriever, dense_retriever, hybrid_retriever = initialize_retrievers(merged_file)

        # Create and launch the Gradio UI
        demo = create_ui(bm25_retriever, dense_retriever, hybrid_retriever)
        demo.css = """
        #title { text-align: center; font-size: 18px; margin-bottom: 20px; }
        #api_key, #query { width: 48%; font-size: 14px; }
        #method label { font-size: 14px; }
        #search_button { font-size: 16px; background-color: #4CAF50; color: white; padding: 10px 20px; }
        #response_box, #sources_box { font-size: 14px; height: 120px; margin-top: 0px; }
        """
        demo.launch()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()