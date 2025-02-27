# Conversational RAG App with Memory 🚀

This project is a Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, and Hugging Face Transformers. It allows users to interact with a conversational AI that retrieves information from a blog post and generates context-aware responses. The app also includes a memory system to retain conversation history, making it feel like a natural chat.


[![Demo Video]()](/imgs,%20vid/Screen%20Recording%202025-02-27%20015204%20(online-video-cutter.com).mp4)

## Overview

This app uses Retrieval-Augmented Generation (RAG) to answer user questions based on the content of a blog post. The blog post used in this project is [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), which discusses the components and challenges of building autonomous agents powered by large language models (LLMs).

**The app is built with:**

- Streamlit: For the user interface.

- LangChain: For the RAG pipeline and conversational memory.

- Hugging Face Transformers: For embeddings and language models.

- BeautifulSoup (bs4): For parsing the blog post.

## Features

1. **Conversational Interface**: Users can ask questions and receive answers in a chat-like format.

2. **Memory System**: The app retains conversation history, allowing for context-aware responses.

3. **Sidebar Settings**: Users can adjust the model's behavior (e.g., temperature, max tokens) to control response randomness and length.

4. **Retrieval-Augmented Generation**: The app retrieves relevant information from the blog post and generates accurate, concise answers.

5. **Customizable**: The app can be easily adapted to work with other documents or datasets.

## How It Works

![alt text](/imgs,%20vid/image.png)

1. Document Loading and Chunking:

    The blog post is loaded and split into smaller chunks using WebBaseLoader and RecursiveCharacterTextSplitter.

    These chunks are stored in a vector store for efficient retrieval.

2. Retrieval Tool:

    A retrieval tool is defined to fetch relevant chunks from the vector store based on the user's query.

3. Conversational Graph:

    A state graph is built using StateGraph and MessagesState to handle the flow of messages between the user, retrieval tool, and response generation.

4. Memory System:

    The conversation history is stored in st.session_state.messages, allowing the app to maintain context across interactions.

5. Sidebar Settings:

    Users can adjust the model's temperature and max tokens to control the randomness and length of responses.

![alt text](/imgs,%20vid/image-1.png)

6. Response Generation:

    The app uses the retrieved information and conversation history to generate context-aware responses.


## Installation
**Prerequisites**

- Python 3.8 or higher
- Streamlit
- LangChain
- LangGraph
- Hugging Face Transformers
- BeautifulSoup (bs4)

### Steps
1. Clone the repository:

```bash
git clone https://github.com/Xmen3em/End-to-End-Building-A-Retrieval-Augmented-Generation-App.git
cd End-to-End-Building-A-Retrieval-Augmented-Generation-App
```

2. Install the required dependencies:

```bash
pip install streamlit langchain langchain-huggingface langchain-core langchain-community bs4
```

or

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the provided URL in your browser to interact with the app.

## Usage
**Ask Questions:**

    Type your question in the chat input box and press Enter.

    The app will retrieve relevant information from the blog post and generate a response.

**Adjust Settings:**

    Use the sidebar to adjust the model's temperature and max tokens.

    Temperature controls the randomness of responses (higher values = more creative, lower values = more deterministic).

    Max tokens limits the length of the response.

**View Conversation History:**

    The chat interface displays the conversation history, allowing you to see previous interactions.

## Customization
**Change the Blog Post**
To use a different blog post or document:

1. Update the web_paths parameter in the WebBaseLoader initialization:

```python
loader = WebBaseLoader(
    web_paths=("https://example.com/your-blog-post",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
```

2. Adjust the chunk_size and chunk_overlap parameters in the RecursiveCharacterTextSplitter to suit your document.

**Modify the Model**
To use a different language model:

1. Update the init_chat_model function:

```python
llm = init_chat_model(
    "your-model-name",
    model_provider="your-model-provider",
    temperature=temperature,
    max_tokens=max_tokens,
)
```

2. Ensure the model is compatible with LangChain and Hugging Face Transformers.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

- Fork the repository.

- Create a new branch for your feature or bug fix.

- Commit your changes and push to the branch.

- Submit a pull request.

## Acknowledgments

- [LangChain](https://langchain.com/) for the RAG pipeline and conversational memory.

- [Hugging Face Transformers](https://huggingface.co/transformers/) for embeddings and language models.

- [Streamlit](https://streamlit.io/) for the user interface.

- [Lilian Weng](https://lilianweng.github.io/) for the blog post used in this project.