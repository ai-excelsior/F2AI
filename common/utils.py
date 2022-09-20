def remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]


def get_default_value():
    return None


def schema_to_dict(schema):
    return {item["name"]: item.get("dtype", "string") for item in schema}
