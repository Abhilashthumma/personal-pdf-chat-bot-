from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_chunks(text):
    """
    Split text into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    return chunks