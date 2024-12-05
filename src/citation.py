from llm import generate_response
from collections import OrderedDict

def generate_citations(response, results, api_key=None, use_llm=True):
    response_with_citations = response
    retrieved_docs = []

    # Generate source references
    for i, result in enumerate(results):
        chunk = result[0]
        source_name = chunk.get('book', chunk.get('source', 'Unknown Source'))
        author = chunk.get('author', 'Unknown Author')
        citation_entry = f"Book [#{i + 1}] {source_name} by {author}"
        retrieved_docs.append(citation_entry)

        # Add citations inline to the response
        text_snippet = chunk['text'][:100]  # Use the first 100 characters for identification
        placeholder = f"[{i + 1}]"

        # Ensure text snippet is found in the response
        if text_snippet in response:
            response_with_citations = response_with_citations.replace(
                text_snippet,
                f"{text_snippet} {placeholder}"
            )
        else:
            # Append the placeholder at the end if snippet not found
            response_with_citations += f" {placeholder}"

    # Remove duplicate sources while preserving order
    unique_docs = list(OrderedDict.fromkeys(retrieved_docs))

    # Generate a textual description of the sources
    sources_with_description = ""
    if use_llm:
        docs_list = "\n".join(unique_docs)
        llm_query = (
            f"Based on the following list of sources, generate a detailed description of their usage in the response:\n"
            f"{docs_list}\n\n"
            "Example:\n"
            "'For the answer, I used the following sources:\n"
            "Book [1]: 07 Harry Potter and the Deathly Hallows.txt by J.K. Rowling, which served as the primary source of information for the initial question,\n"
            "and Book [4]: 01 Harry Potter and the Sorcerer's Stone.txt by J.K. Rowling, which provided supporting details for the first part of the answer.'"
        )
        try:
            # Generate detailed descriptions using LLM
            sources_with_description = generate_response(
                query=llm_query,
                context_chunks=[],
                api_key=api_key
            )
        except Exception as e:
            # Fallback to a simple list of sources if an error occurs
            sources_with_description = (
                f"An error occurred while generating detailed citations: {e}. "
                f"Sources used: {', '.join(unique_docs)}."
            )
    else:
        # Simple fallback if LLM is not used
        sources_with_description = f"For the answer, I used the following sources: {', '.join(unique_docs)}."

    return response_with_citations, "\n".join(unique_docs), sources_with_description