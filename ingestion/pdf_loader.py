import pymupdf

def looks_like_research_paper(pages):

    text = " ".join(
        page["text"] for page in pages
    ).lower()

    signals = [
        "abstract",
        "introduction",
        "references",
        "keywords",
        "methodology"
    ]

    score = sum(
        1 for signal in signals
        if signal in text
    )

    return score >= 3

def load_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })
    doc.close()

    return pages

if __name__ == "__main__":
    pdf_path = "data/raw/2410.14077v2.pdf"

    pages = load_pdf(pdf_path)

    print("Number of pages:", len(pages))

    print("\nFirst page:")
    print(pages[0]["text"][:1000])