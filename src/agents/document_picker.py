import json
import logging

from llm_adapters import model_config
from llm_adapters.llm_adapter import LLMClientFactory

logger = logging.getLogger(__name__)


class PickerModel(model_config.ModelConfig):
    MODEL = 'gpt-4o'
    MAX_TOKENS = 300
    CONTEXT_WINDOW = 8192
    TEMPERATURE = 0.4  # Keep outputs deterministic for scoring and ranking
    RESPONSE_TOKENS = 100  # Ensure response fits within limits
    FREQUENCY_PENALTY = 0.1  # Avoid repetition in rationale
    PRESENCE_PENALTY = 0  # Encourage new ideas or highlighting unique patterns


system_prompt_test = """
You are a procurement document analysis expert.
Your task is to identify key documents that contain specific tender assignment details ('zadání').
Focus exclusively on documents with actual tender specifications and requirements in Czech language.
Based on the provided list of documents, find those containing detailed tender assignments.
Look specifically for documents containing 'zadání', 'zadávací dokumentace', 
detailed technical specifications and requirement descriptions.
Strictly filter out:
- Blank or unfilled forms
- Generic statements and declarations
- Administrative documents without specific tender details
- Routine statements and standard templates
Return only the IDs of documents that contain concrete tender assignment information.
Use function filter_important_documents for response formating."""

user_prompt_test = '{documents}'


def pick_important_documents(document_list):
    llm_client = LLMClientFactory.create_llm_client(PickerModel)
    user_prompt = user_prompt_test.format(documents=document_list)
    funcs = [
        {
            "name": "filter_important_documents",
            "description": "Filter and return IDs of important procurement-related documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of document IDs that are considered important for public procurement"
                    }
                },
                "required": ["document_ids"]
            }
        }
    ]

    response = llm_client.call_with_functions(
        system_prompt=system_prompt_test,
        user_prompt=str(user_prompt),
        functions=funcs
    )
    logger.info(f'response: {response}')
    parsed_output = response.choices[0].message.function_call.arguments
    structured_data = json.loads(parsed_output)
    logger.info(f"agent call success")
    return structured_data
