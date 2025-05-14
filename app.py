import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import bs4

# Streamlit app setup
st.title("ISET Sfax Chatbot")
st.markdown("Ask questions about ISET Sfax (e.g., programs, departments, or director).")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Secure API keys
# os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
google_api_key = "AIzaSyApH8Sk8uPWe1NS0Ie98uDQD3jo89VD_UA"


# Cache RAG pipeline
@st.cache_resource
def load_rag_pipeline():
    # Load documents
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

    # Fallback document for missing director info
    blog_docs.append(
        Document(
            page_content="The director of ISET Sfax is not explicitly named in the provided sources. A possible contact is Mr Ahmed Jmal (unconfirmed). Email isetsf@isetsf.net for the latest information.",
            metadata={"source": "manual"},
        )
    )

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    splits = text_splitter.split_documents(blog_docs)

    # Create/load vector store
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

    # Define LLM (switch to Grok if desired)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", temperature=0, google_api_key=google_api_key
    )
    # Alternative: llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)

    # Set up prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Act as a chatbot for ISET Sfax, answering based only on the following context: {context}",
            ),
            ("human", "{input}"),
        ]
    )

    # Create chain
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# Load pipeline
rag_chain = load_rag_pipeline()

# Chat interface
with st.form("chat_form"):
    query = st.text_input(
        "Your question:", placeholder="What is the name of the director of ISET Sfax?"
    )
    submit = st.form_submit_button("Ask")

if submit and query:
    try:
        with st.spinner("Generating response..."):
            response = rag_chain.invoke({"input": query})
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
