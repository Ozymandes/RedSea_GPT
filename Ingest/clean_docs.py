def filter_pages(docs, min_chars=300):
    return [
        d for d in docs
        if len(d.page_content.strip()) >= min_chars
    ]
