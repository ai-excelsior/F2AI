def remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]
