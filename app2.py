import streamlit as st
import os
import pandas as pd
from datetime import datetime
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import bs4

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stChatMessage.user { 
        background-color: #e0f7fa; 
        border-radius: 10px; 
        padding: 10px; 
        margin: 5px 0; 
        max-width: 70%; 
        align-self: flex-end; 
    }
    .stChatMessage.assistant { 
        background-color: #f1f8e9; 
        border-radius: 10px; 
        padding: 10px; 
        margin: 5px 0; 
        max-width: 70%; 
        align-self: flex-start; 
    }
    .chat-container { 
        display: flex; 
        flex-direction: column; 
        gap: 10px; 
        max-height: 60vh; 
        overflow-y: auto; 
        padding: 10px; 
        border: 1px solid #ddd; 
        border-radius: 10px; 
    }
    .header { 
        text-align: center; 
        padding: 20px; 
        background-color: #007bff; 
        color: white; 
        border-radius: 10px; 
        margin-bottom: 20px; 
    }
    .sidebar-content { 
        font-size: 14px; 
        line-height: 1.6; 
    }
    .feedback-btn { 
        margin: 0 5px; 
        font-size: 20px; 
        cursor: pointer; 
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Streamlit app setup
st.markdown(
    '<div class="header"><h1>ISET Sfax Chatbot</h1><p>Ask about programs, departments, or the director of ISET Sfax</p></div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("About")
    st.write(
        "This chatbot provides information about ISET Sfax based on its official website. Contact isetsf@isetsf.net for more details."
    )
    st.write("[Visit ISET Sfax](https://isetsf.rnu.tn/fr)")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = {}

# Load secrets
try:
    # os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    # os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    # google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = (
        "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"
    )
    google_api_key = "AIzaSyApH8Sk8uPWe1NS0Ie98uDQD3jo89VD_UA"
except KeyError as e:
    st.error(
        f"Missing secret: {e}. Please add it to .streamlit/secrets.toml or set it as an environment variable."
    )
    st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {str(e)}")
    st.stop()

# Set LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


# Cache RAG pipeline
@st.cache_resource
def load_rag_pipeline():
    try:
        loader = WebBaseLoader(
            web_paths=[
                "https://isetsf.rnu.tn/fr",
                "https://isetsf.rnu.tn/fr/categories/27/licence-nationale",
                "http://www.isetjb.rnu.tn/fr/institut/reseau-iset-tunisie.html",
                "https://isetsf.rnu.tn/fr/institut/presentation",
                "https://isetsf.rnu.tn/fr/institut/organigramme",
                "https://isetsf.rnu.tn/fr/institut/associations/association-sportive-universitaire-iset-de-sfax-asuis",
                "https://isetsf.rnu.tn/fr/institut/associations/association-de-recherche-et-developpement-et-d-innovation-ardi",
                "https://isetsf.rnu.tn/fr/institut/departements/sc-eco-et-gestion",
                "https://isetsf.rnu.tn/fr/institut/departements/technologies-de-l-informatique",
                "https://isetsf.rnu.tn/fr/institut/departements/genie-des-procedes",
                "https://isetsf.rnu.tn/fr/institut/departements/genie-civil",
                "https://isetsf.rnu.tn/fr/institut/departements/genie-mecanique",
                "https://isetsf.rnu.tn/fr/article/528/loi-de-creation",
            ],
            bs_kwargs={"parse_only": bs4.SoupStrainer(["div", "p", "h1", "h2", "h3"])},
        )
        blog_docs = loader.load()
    except Exception as e:
        blog_docs = [
            Document(
                page_content=f"Error loading documents: {str(e)}",
                metadata={"source": "error"},
            )
        ]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    splits = text_splitter.split_documents(blog_docs)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            persist_directory="./chroma_db", embedding_function=embedding_model
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embedding_model, persist_directory="./chroma_db"
        )
    vectorstore.persist()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", temperature=0, google_api_key=google_api_key
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Act as a chatbot for ISET Sfax, answering based only on the following context: {context}",
            ),
            ("human", "{input}"),
        ]
    )

    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# Load pipeline
rag_chain = load_rag_pipeline()

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if (
            msg["role"] == "assistant"
            and idx not in st.session_state.feedback_submitted
        ):
            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍", key=f"thumbs_up_{idx}", help="Thumbs Up"):
                    save_feedback(msg["query"], msg["content"], "thumbs_up", "")
                    st.session_state.feedback_submitted[idx] = True
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎", key=f"thumbs_down_{idx}", help="Thumbs Down"):
                    feedback = st.text_area(
                        "Optional feedback:", key=f"feedback_text_{idx}"
                    )
                    if st.button("Submit Feedback", key=f"submit_feedback_{idx}"):
                        save_feedback(
                            msg["query"], msg["content"], "thumbs_down", feedback
                        )
                        st.session_state.feedback_submitted[idx] = True
                        st.success("Thank you for your feedback!")
st.markdown("</div>", unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask about ISET Sfax (e.g., director, programs):")
if query:
    st.session_state.messages.append({"role": "user", "content": query, "query": query})
    with st.spinner("Generating response..."):
        try:
            response = rag_chain.invoke({"input": query})
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"], "query": query}
            )
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


# Feedback storage
def save_feedback(query, response, rating, comments):
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "response": response,
        "rating": rating,
        "comments": comments,
    }
    df = pd.DataFrame([feedback_data])
    feedback_file = "feedback.csv"
    if os.path.exists(feedback_file):
        df.to_csv(feedback_file, mode="a", header=False, index=False)
    else:
        df.to_csv(feedback_file, index=False)


# Admin feedback view (password-protected)
with st.sidebar:
    st.header("Admin Access")
    password = st.text_input("Enter admin password:", type="password")
    if password == "iset_admin_2025":  # Replace with a secure password
        st.subheader("Feedback Summary")
        if os.path.exists("feedback.csv"):
            df = pd.read_csv("feedback.csv")
            st.write(f"Total feedback entries: {len(df)}")
            st.write(
                f"Average rating: {df['rating'].value_counts().get('thumbs_up', 0) / len(df):.2%} positive"
            )
            st.dataframe(df[["timestamp", "query", "rating", "comments"]])
        else:
            st.write("No feedback yet.")
