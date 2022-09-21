def remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]


def get_default_value():
    return None


def schema_to_dict(schema):
    return {item["name"]: item.get("dtype", "string") for item in schema}


def service_to_dict(schema):
    item_dict = {}
    for item in schema:
        split_item = item.split(":")
        if len(split_item) == 2 or len(split_item) == 3:
            if split_item[0] not in item_dict:
                item_dict.update(
                    {
                        split_item[0]: [
                            {
                                (split_item[1] if split_item[1] != "*" else "__all__"): None
                                if len(split_item) == 2
                                else split_item[2]
                            }
                        ]
                    }
                )
            else:
                item_dict[split_item[0]].append(
                    {
                        (split_item[1] if split_item[1] != "*" else "__all__"): None
                        if len(split_item) == 2
                        else split_item[2]
                    }
                )
        else:
            raise ValueError("Please make sure colon not in name of table or features")
    return item_dict
