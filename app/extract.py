from PyPDF2 import PdfReader

def extract_text(file):
    """
        :param file: the PDF file to extract
    """
    content = ""
    reader = PdfReader(file)
    number_of_pages = len(reader.pages)

    # Scrape text from multiple pages
    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        content = content + text

    return content

