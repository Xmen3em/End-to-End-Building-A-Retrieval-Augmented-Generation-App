import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import bs4
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

# Initialize the chat model and embeddings
def initialize_model(temperature, max_tokens):
    """Initialize the chat model with user-defined settings."""
    return init_chat_model(
        "qwen-2.5-32b",
        model_provider="groq",
        temperature=temperature,
        max_tokens=max_tokens,
    )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store.add_documents(all_splits)

# Define the retrieval tool
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    '''Retrieve Information related to a query'''
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = '\n\n'.join((f'Source: {doc.metadata}\nContent: {doc.page_content}') for doc in retrieved_docs)
    return serialized, retrieved_docs

# Define the state for the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define the graph builder
graph_builder = StateGraph(MessagesState)

# Step 1: Generate an AI message that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Step 2: Execute the retrieval
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content
def generate(state: MessagesState):
    """Generate answers"""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == 'tool':
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    
    # Format into prompt
    docs_content = '\n\n'.join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversational_messages = [
        message for message in state["messages"] if message.type in ('human', 'system') or (message.type == 'ai' and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversational_messages
    
    # Generate response
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Add nodes to the graph
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

# Set entry point and conditional edges
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

# Compile the graph
graph = graph_builder.compile()

# Streamlit App
st.title("Conversational RAG App with Memory 🤖")

# Sidebar for additional features
with st.sidebar:
    st.header("Settings ⚙️")
    st.markdown("### Configure the RAG App")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, help="Controls randomness in the model's responses.")
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=200, help="Limits the length of the response.")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses Retrieval-Augmented Generation (RAG) to answer questions based on a blog post.")
    st.markdown("Built with 🚀 by Menem")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the model with sidebar settings
llm = initialize_model(temperature, max_tokens)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a question about the blog post:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the input through the graph
    response = graph.invoke({"messages": st.session_state.messages})
    ai_response = response["messages"][-1].content

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)