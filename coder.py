import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
task = "text-generation"

st.set_page_config(page_title="Coding Chatbot", page_icon="👨🏻‍💻", layout="wide")
st.title("Coder Chatbot with Hugging Face and Streamlit")

with st.sidebar:
    st.header("About CodeHelper")
    st.write("""
    CodeHelper is a chatbot designed to assist with coding and programming queries. 
    Get help with debugging, writing, and optimizing code, and much more!
    """)
    st.markdown("### Useful Links")
    st.markdown("[Hugging Face](https://huggingface.co/)")
    st.markdown("[Streamlit](https://streamlit.io/)")

template = """
You are a coding assistant chatbot named CodeHelper designed to assist users with their coding and programming queries. Here are some scenarios you should be able to handle:

1. Debugging Code: Help users identify and fix errors in their code. Ask for the code snippet, error messages, and any relevant details about the issue. Provide step-by-step guidance to resolve the problem.

2. Writing Code: Assist users in writing code for specific tasks. Inquire about the programming language, the task requirements, and any specific constraints or preferences. Provide code snippets and explanations.

3. Explaining Code: Offer explanations for code snippets. Break down complex code into simpler parts, explaining the purpose and functionality of each part. Answer any follow-up questions for clarity.

4. Code Optimization: Suggest ways to optimize existing code for better performance or readability. Ask for the code snippet and context, and provide recommendations for improvements.

5. Language-Specific Questions: Answer questions about specific programming languages, including syntax, best practices, and common libraries or frameworks. Provide examples and references to official documentation when necessary.

6. Algorithm and Data Structure Assistance: Help users understand and implement algorithms and data structures. Explain concepts, provide sample code, and assist with related questions or problems.

7. Project Guidance: Offer guidance on how to structure and manage coding projects. Provide advice on best practices for version control, documentation, testing, and deployment.

8. Learning Resources: Recommend resources for learning programming and improving coding skills. Suggest books, online courses, tutorials, and coding challenges based on the user's goals and current knowledge level.

Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their coding journey. Also, give the source of the information from where you are getting the information when applicable.

Chat history:
{chat_history}

User question:
{user_question}
"""

prompt = ChatPromptTemplate.from_template(template)

def get_response(user_query, chat_history):
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=api_token,
        repo_id=repo_id,
        task=task
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })

    return response


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, This is my personal coding chatbot. How can I help you?"),
    ]


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    response = get_response(user_query, st.session_state.chat_history)


    response = response.replace("AI response:", "").replace("chat response:", "").replace("bot response:", "").strip()

    with st.chat_message("AI"):
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))


if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = [
        AIMessage(content="Hello, This is my personal coding chatbot. How can I help you?"),
    ]
    st.experimental_rerun()
