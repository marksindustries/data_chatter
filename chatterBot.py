import streamlit as st
import pandas as pd
import json
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from jinja2 import Environment, BaseLoader
from dotenv import load_dotenv
import os
import io

# Load environment variables and set up Groq
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
llm = ChatGroq(model_name="mixtral-8x7b-32768")

# Sidebar for file upload
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load data


@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        return pd.read_csv("orders.csv")  # Default file if none uploaded


if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    data = load_data(None)

# Set up the Jinja template for LangChain
template = """
Please generate a Jinja template to answer the question using pandas operations on DataFrames. Your response should ONLY be based on the given context and follow the response guidelines and format instructions.
===Tables
{list_of_table_columns}
===Response Guidelines
1. If the provided context is sufficient, please generate a valid Jinja template without any explanations for the question. The template should start with a comment containing the question being asked.
2. Use pandas operations on the DataFrame 'data' to answer the question.
3. Do not use SQL queries. Instead, use pandas methods like .mean(), .sum(), .groupby(), etc.
5. If the provided context is insufficient, please explain why it can't be generated.
6. Please use the DataFrame named 'data'.
7. Please format the template before responding.
8. Please always respond with a valid well-formed JSON object with the following format
===Response Format
{{
 "template": "Generated Jinja template when context is sufficient.",
 "explanation": "An explanation of failing to generate the template."
}}
===Question
{question}
"""

prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | llm

# Helper functions


def parse_json_response(response_content):
    cleaned_content = response_content.strip()
    if cleaned_content.startswith('```') and cleaned_content.endswith('```'):
        cleaned_content = cleaned_content[3:-3]
    cleaned_content = cleaned_content.strip()
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        cleaned_content = cleaned_content.replace('\n', '').replace('\r', '')
        return json.loads(cleaned_content)


def render_jinja_template(jinja_template, **kwargs):
    jinja_template = jinja_template.replace(
        '{% comment %}', '{#').replace('{% endcomment %}', '#}')
    env = Environment(loader=BaseLoader())
    env.globals.update(kwargs)
    template = env.from_string(jinja_template)
    return template.render()


# Streamlit UI
st.title("Data Chatter")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about the data?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response using LangChain and Jinja
    response = chain.invoke({
        "question": prompt,
        "list_of_table_columns": data.columns.tolist(),
    })

    try:
        result = parse_json_response(response.content)
        if "template" in result:
            rendered_result = render_jinja_template(
                result["template"], data=data, pd=pd)
            response_content = f"Analysis result:\n\n{rendered_result}"

            if "explanation" in result:
                response_content += f"\n\nExplanation: {result['explanation']}"
        else:
            response_content = result.get(
                "explanation", "Unable to generate a response.")
    except Exception as e:
        response_content = f"Error processing response: {str(e)}\n\nRaw response:\n{response.content}"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_content)
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content})


with st.sidebar:
    # Display data preview
    if st.checkbox("Show data preview"):
        st.write(data.head())

    # Display column information
    if st.checkbox("Show column information"):
        st.write(data.dtypes)
