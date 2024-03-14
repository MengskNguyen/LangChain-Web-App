import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import dotenv

dotenv.load_dotenv()

template = """
    Below is an email that may be poorly worded.
    Your goal is to:
    - Properly format the email
    - Convert the input text to a specified tone
    - Convert the input text to a specified dialect
    
    Here are some examples different tones:
    - Formal: We went to Barcelona for the weekend. We have a lot of things to tell you
    - Informal: Went to Barcelona this weekend. Lots to tell you
    
    Here are some examples of words in different dialects:
    - American English: French Fries, cotton candy, apartment
    - British English: chips, candyfloss, flag
    
    Below is the email, tone, and dialect:
    TONE: {tone}
    DIALECT: {dialect}
    EMAIL: {email}
    
    YOUR RESPONSE:
"""
prompt = PromptTemplate.from_template(template)


def load_LLM():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0.5)
    return llm


llm = load_LLM()
parser = StrOutputParser()
st.set_page_config(page_title="Globalize email", page_icon=':robot:')
st.header("Globalize email")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Transform your email into professional format with Langchain")

with col2:
    st.image(image="deer.png", width=350, caption="A deer")

st.markdown("## Enter Your Email To Convert")

col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        "Which tone would you like your email to have?",
        ('formal', 'Informal')
    )

with col2:
    option_dialect = st.selectbox(
        "Which English dialect would you like?",
        ('American English', 'British English')
    )


def get_text():
    input_text = st.text_area(label="", placeholder="Your Email ...", key="email_input")
    return input_text


email_input = get_text()

st.markdown("## Your Convert Email:")

if email_input:
    chain = ({'tone': RunnablePassthrough(), 'dialect': RunnablePassthrough(),
              'email': RunnablePassthrough()} | prompt | llm | parser)
    output = chain.invoke({'tone': option_tone, 'dialect': option_dialect, 'email': email_input})

    st.write(output)
