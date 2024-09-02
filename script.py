import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
import streamlit as st
import typing_extensions as typing
import logging
import json
import os
from dotenv import load_dotenv

# Define TypedDict for Gemini response
class GeminiResponse(typing.TypedDict):
    rule_name: str
    label: bool
    part: list[str]
    suggestion: list[str]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()
GOOGLE_API_KEY = st.secrets["API_KEY"]

# Configure Google Generative AI with API Key
genai.configure(api_key=GOOGLE_API_KEY)

# Define Generation and Safety Configurations
genai_generation_config = GenerationConfig(
    candidate_count=1,
    max_output_tokens=800,
    temperature=0,
    response_mime_type="application/json",
    response_schema=GeminiResponse
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

def gemini_answer(prompt: str, model: genai.GenerativeModel) -> typing.Optional[str]:
    """Generate content using the Gemini model and return the response text."""
    try:
        response = model.generate_content(prompt, generation_config=genai_generation_config, safety_settings=safety_settings)
        response_text = response.parts[0].text
        logger.info(f"Response: {response_text}")
        return response_text
    except json.JSONDecodeError:
        logger.error("Invalid JSON output string")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


def create_rules_list(input_string: str):
    seperator = '##'
    return input_string.split(seperator)


def inference(system_message: str, model_name: str, rules_list: list[str], sales_deck: str) -> typing.Optional[str]:
    """Perform inference using the Gemini model and return the generated response."""
    output_list=[]

    for rule in rules_list:
        input_text = f"""
        The rule is: {rule}
        The sales deck to evaluate is: {sales_deck}
        """ 

        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_message)
        model_output = gemini_answer(input_text, model)
        output_list.append(model_output)

    return output_list
