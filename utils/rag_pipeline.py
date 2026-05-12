from langchain_community.llms import Ollama


def get_answer(vector_store, question):
    """
    Retrieve relevant chunks and generate answer.
    """

    # Search similar chunks
    docs = vector_store.similarity_search(
        question,
        k=3
    )

    # Combine chunk text
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
    You are a helpful AI assistant.

    Answer ONLY from the provided context.

    If answer is not found in context, say:
    "Answer not found in uploaded documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Load Ollama model
    llm = Ollama(model="mistral")

    # Generate response
    response = llm.invoke(prompt)

    return response, docs