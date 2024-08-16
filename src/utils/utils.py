import hashlib


def get_short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def get_sha512_hash_hex(string: str) -> str:
    hash_object = hashlib.sha512(string.encode('utf-8'))
    return hash_object.hexdigest()
