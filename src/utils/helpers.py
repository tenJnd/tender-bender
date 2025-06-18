import logging
from dataclasses import fields, is_dataclass
from typing import TypeVar, Dict, Any, List, Set, Union
from typing import get_origin, get_args

import tiktoken
from llm_adapters import model_config

logger = logging.getLogger(__name__)

T = TypeVar('T')


def filter_dataclass(obj: T, include: Union[List[str], Set[str]] = None,
                     exclude: Union[List[str], Set[str]] = None) -> Dict[str, Any]:
    """
    Filter a dataclass object to include or exclude specific fields.

    Args:
        obj: A dataclass object
        include: List of field names to include (if None, includes all fields not in exclude)
        exclude: List of field names to exclude (if None, excludes no fields)

    Returns:
        Dictionary with only the requested fields
    """
    if include is not None and exclude is not None:
        raise ValueError("Only one of 'include' or 'exclude' should be provided")

    data = asdict(obj)

    if include is not None:
        # Include only specified fields
        include_set = set(include)
        return {k: v for k, v in data.items() if k in include_set}

    if exclude is not None:
        # Exclude specified fields
        exclude_set = set(exclude)
        return {k: v for k, v in data.items() if k not in exclude_set}

    # If neither include nor exclude is specified, return all fields
    return data


def get_tokenizer_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


def chunk_text_for_user_prompt(
        m_config: model_config.ModelConfig,
        system_prompt: str,
        user_prompt_template: str,
        full_text: str
) -> List[str]:
    """
    Splits `full_text` into token-safe chunks that can be inserted into the user prompt,
    such that system_prompt + user_prompt(chunked) + response_tokens <= context_window.
    """
    tokenizer = get_tokenizer_for_model(m_config.MODEL)

    # Reserve system + response
    system_tokens = count_tokens(system_prompt, tokenizer)
    draft_user_prompt_tokens = count_tokens(user_prompt_template, tokenizer)
    response_tokens = m_config.RESPONSE_TOKENS
    context_limit = m_config.CONTEXT_WINDOW
    max_available_tokens = context_limit - system_tokens - response_tokens - draft_user_prompt_tokens

    if max_available_tokens <= 0:
        raise ValueError("Not enough room for document text. Try shortening your prompt.")

    logger.info(f"System prompt: {system_tokens} tokens")
    logger.info(f"User prompt (draft): {draft_user_prompt_tokens} tokens")
    logger.info(f"Available for document content: {max_available_tokens} tokens")

    # Chunk the full_text based on usable token budget
    paragraphs = full_text.split("\n\n")
    chunks = []
    current_chunk = ""
    token_count = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, tokenizer)
        if token_count + para_tokens > max_available_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
            token_count = para_tokens
        else:
            current_chunk += "\n\n" + para
            token_count += para_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def _get_properties_and_required(cls) -> (Dict[str, Any], List[str]):
    """
    Helper function to generate properties and required list for a dataclass.
    Handles nested dataclasses recursively.
    """
    properties = {}
    required = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    for field in fields(cls):
        name = field.name
        field_type = field.type
        origin = get_origin(field_type)

        # Determine if field is Optional
        if origin is Union and type(None) in get_args(field_type):
            actual_type = [t for t in get_args(field_type) if t is not type(None)][0]
            is_optional = True
        else:
            actual_type = field_type
            is_optional = False

        field_schema = {}

        if is_dataclass(actual_type):
            # Recursively handle embedded dataclasses
            nested_properties, nested_required = _get_properties_and_required(actual_type)
            field_schema["type"] = "object"
            field_schema["properties"] = nested_properties
            field_schema["required"] = nested_required
        elif origin is list:
            item_type = get_args(actual_type)[0]
            field_schema["type"] = "array"
            item_schema = {}
            if is_dataclass(item_type):
                nested_properties, nested_required = _get_properties_and_required(item_type)
                item_schema["type"] = "object"
                item_schema["properties"] = nested_properties
                item_schema["required"] = nested_required
            else:
                item_schema["type"] = type_map.get(item_type, "string")
            field_schema["items"] = item_schema
        elif origin is dict:  # Handle dictionary types (basic object)
            field_schema["type"] = "object"
            field_schema["additionalProperties"] = True # Added this line for arbitrary dicts
        else:
            # Basic types
            field_schema["type"] = type_map.get(actual_type, "string")

        # Add description if present
        if "description" in field.metadata:
            field_schema["description"] = field.metadata["description"]

        properties[name] = field_schema

        if not is_optional:
            required.append(name)

    return properties, required


def dataclass_to_openai_schema(cls, function_name: str, function_desc: str):
    """
    Converts a Python dataclass into an OpenAI function call schema.
    Handles nested dataclasses and lists of dataclasses.
    """
    properties, required = _get_properties_and_required(cls)

    return {
        "name": function_name,
        "description": function_desc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def empty_dict_from_dataclass(cls) -> dict:
    """
    Recursively constructs a nested dictionary with default empty values
    based on the given dataclass schema.
    """
    result = {}

    for field in fields(cls):
        typ = field.type
        origin = get_origin(typ)

        # Unwrap Optional[X] to get base type
        if origin is Union and type(None) in get_args(typ):
            base_type = [t for t in get_args(typ) if t is not type(None)][0]
        else:
            base_type = typ

        base_origin = get_origin(base_type)

        if is_dataclass(base_type):
            result[field.name] = empty_dict_from_dataclass(base_type)
        elif base_origin == list:
            result[field.name] = []
        elif base_origin == dict:
            result[field.name] = {}
        elif base_type == bool:
            result[field.name] = False
        else:
            result[field.name] = None

    return result
