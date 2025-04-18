import streamlit as st
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector
from textwrap import dedent
import time
import os

pg_pass = st.secrets["PG_PASS"]

db_url = f"postgresql+psycopg2://postgres:{pg_pass}@database-1.czg44aga0cfb.ap-south-1.rds.amazonaws.com:5432/ai"

st.set_page_config(page_title="Virtual Assistant", page_icon="🤖", layout="wide")
st.title("Virtual Assistant")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'knowledge_base_initialized' not in st.session_state:
    knowledge_base = WebsiteKnowledgeBase(
        urls=["https://shapefy.in"],
        max_links=700,
        vector_db=PgVector(
            table_name="shapefy_kb",
            db_url=db_url,
            embedder=GeminiEmbedder(),
        ),
    )
    # knowledge_base.load(recreate=True)
    st.session_state['knowledge_base'] = knowledge_base
    st.session_state['knowledge_base_initialized'] = True

def build_conversation_context(messages):
    if not messages:
        return ""

    context = "Previous conversation:\n"
    for msg in messages:
        prefix = "User: " if msg["role"] == "user" else "Assistant: "
        context += f"{prefix}{msg['content']}\n\n"

    return context

def get_agent_with_context():
    if len(st.session_state['messages']) > 0 and st.session_state['messages'][-1]['role'] == 'user':
        history = st.session_state['messages'][:-1]
    else:
        history = st.session_state['messages']

    context = build_conversation_context(history)

    return Agent(
        model=Gemini(id="gemini-2.0-flash"),
        description="""
        You are an AI Agent.
        Your goal is to provide information from the vector DB.
        """,
        instructions=dedent(f"""
        1. Analyze the request.
        2. Search your knowledge base for relevant information.
        3. Present the information to the user.
        4. Provide concise, detailed but accurate answers based on the context.
        5. Do not make up or infer information that is not in the context.
        6. If the information needed is not available in the provided context, respond with "I don't have enough information to answer this question accurately."
        7. Maintain a conversational tone and refer to previous parts of the conversation when relevant.
        8. Remember details that the user has shared previously.

        {context}
        """),
        knowledge=st.session_state['knowledge_base'],
    )

for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message("user").markdown(message['content'])
    else:
        st.chat_message("assistant").markdown(message['content'])

def handle_input():
    question = st.session_state.input_field
    if question:
        st.session_state['messages'].append({"role": "user", "content": question})

        with st.spinner('Thinking...'):
            contextual_agent = get_agent_with_context()

            response_object = contextual_agent.run(question, markdown=True)
            response_text = response_object.content

        st.session_state['messages'].append({"role": "assistant", "content": response_text})

        st.session_state.input_field = ""

st.text_input(
    "Ask a question:",
    key="input_field",
    placeholder="Type your question here...",
    on_change=handle_input
)

if st.button("Clear Conversation"):
    st.session_state['messages'] = []
    st.rerun()
