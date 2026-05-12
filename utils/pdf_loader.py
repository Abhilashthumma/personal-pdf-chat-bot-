from PyPDF2 import PdfReader


def extract_text_from_pdfs(pdf_files):
    """
    Extract text from uploaded PDF files.
    """

    text = ""

    for pdf in pdf_files:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text