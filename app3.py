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
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time  # Import time for potential delays
import shutil  # Import shutil for directory removal

# Download NLTK data (Corrected Error Handling and added punkt_tab)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:  # Corrected exception
    st.info("Downloading NLTK 'punkt' tokenizer data...")
    nltk.download("punkt")
    st.info("Download complete.")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:  # Corrected exception
    st.info("Downloading NLTK 'wordnet' corpus data...")
    nltk.download("wordnet")
    st.info("Download complete.")
try:
    # Added download for punkt_tab based on your error log
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:  # Corrected exception
    st.info("Downloading NLTK 'punkt_tab' tokenizer data...")
    nltk.download("punkt_tab")
    st.info("Download complete.")


# Custom CSS
st.markdown(
    """
    <style>
    .stChatMessage.user { background-color: #e0f7fa; border-radius: 10px; padding: 10px; margin: 5px 0; max-width: 70%; align-self: flex-end; }
    .stChatMessage.assistant { background-color: #f1f8e9; border-radius: 10px; padding: 10px; margin: 5px 0; max-width: 70%; align-self: flex-start; }
    .chat-container { display: flex; flex-direction: column; gap: 10px; max-height: 60vh; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 10px;}
    .header { text-align: center; padding: 20px; background-color: #007bff; color: white; border-radius: 10px; margin-bottom: 20px; }
    .sidebar-content { font-size: 14px; line-height: 1.6; }
    .feedback-btn { margin: 0 5px; font-size: 20px; cursor: pointer; }
    </style>
""",
    unsafe_allow_html=True,
)

# Streamlit app setup
st.markdown(
    '<div class="header"><h1>ISET Sfax Chatbot</h1><p>Ask about programs, departments, or the director of ISET Sfax</p></div>',
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = {}
if "user_id" not in st.session_state:
    st.session_state.user_id = "guest"  # Replace with login system if needed later
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "RAG"  # Default model

# Load secrets
try:
    # os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"] # Uncomment if using Groq models
    # os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"] # Uncomment if using Langchain tracing
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError as e:
    st.error(
        f"Missing secret: {e}. Please add it to .streamlit/secrets.toml or set it as an environment variable."
    )
    st.stop()  # Stop execution if secrets are missing


# NLTK model setup
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()


def preprocess(text):
    # Ensure text is a string
    if not isinstance(text, str):
        return []
    try:
        tokens = word_tokenize(text.lower())
        return [lemmatizer.lemmatize(token) for token in tokens]
    except Exception as e:
        st.warning(f"NLTK preprocessing failed for text '{text[:50]}...': {e}")
        return []


def load_nltk_dataset():
    dataset_file = "nltk_dataset.csv"
    if os.path.exists(dataset_file):
        try:
            df = pd.read_csv(dataset_file)
            # Ensure required columns exist and handle potential NaNs
            if "question" in df.columns and "answer" in df.columns:
                # Filter out rows with missing questions or answers
                df = df.dropna(subset=["question", "answer"])
                # Convert to string just in case
                return (
                    df["question"].astype(str).tolist(),
                    df["answer"].astype(str).tolist(),
                )
            else:
                st.warning(
                    f"'{dataset_file}' exists but is missing 'question' or 'answer' columns. Contents:\n{df.head()}"
                )
                return [], []
        except Exception as e:
            st.warning(f"Could not load NLTK dataset '{dataset_file}': {e}")
            return [], []
    return [], []


# Load NLTK data and fit vectorizer on app start
# Note: This is only done once. New data from feedback requires app restart to be included.
questions, answers = load_nltk_dataset()
if questions:
    # Preprocess questions before fitting
    processed_questions = [" ".join(preprocess(q)) for q in questions]
    X = vectorizer.fit_transform(processed_questions)
    st.sidebar.info(f"NLTK model loaded with {len(questions)} known Q&A pairs.")
else:
    X = None
    st.sidebar.warning(
        "NLTK dataset is empty. NLTK model will not be able to answer questions."
    )


def nltk_model(query):
    if not questions or X is None:
        return "My NLTK knowledge base is empty. Please try the RAG model or provide feedback on RAG answers to help me learn!"

    # Preprocess and vectorize the query
    processed_query = " ".join(preprocess(query))
    if not processed_query:  # Handle cases where preprocessing yields empty string
        return "I could not process your query for NLTK matching."

    try:
        query_vec = vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, X)

        # Find the best match
        max_similarity = similarities.max()

        # Use a threshold to determine if the match is good enough
        similarity_threshold = 0.6  # Tune this threshold

        if max_similarity > similarity_threshold:
            best_idx = similarities.argmax()
            return answers[best_idx]
        else:
            return "I don't have a specific answer for that in my NLTK knowledge base. Try switching to the RAG model!"
    except Exception as e:
        st.warning(f"Error during NLTK similarity check: {e}")
        return "An error occurred while using the NLTK model."


# RAG pipeline
# Using st.cache_resource to load this only once per session
@st.cache_resource
def load_rag_pipeline(google_api_key):
    # --- Document Loading ---
    web_urls = [
        "https://isetsf.rnu.tn/fr",
        "https://isetsf.rnu.tn/fr/categories/27/licence-nationale",
        # "http://www.isetjb.rnu.tn/fr/institut/reseau-iset-tunisie.html", # Excluding this URL as it's not ISET Sfax
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
    ]
    blog_docs = []
    successful_urls = []
    failed_urls = []

    st.sidebar.info(f"Attempting to load documents from {len(web_urls)} URLs...")
    for url in web_urls:
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={
                    "parse_only": bs4.SoupStrainer(
                        ["div", "p", "h1", "h2", "h3", "article", "section"]
                    )
                },  # Added more tags
            )
            docs = loader.load()
            blog_docs.extend(docs)
            successful_urls.append(url)
        except Exception as e:
            failed_urls.append(url)
            # st.warning(f"Failed to load {url}: {e}") # Too verbose in sidebar

    st.sidebar.info(
        f"Successfully loaded {len(successful_urls)} URLs. Failed to load {len(failed_urls)} URLs."
    )
    if failed_urls:
        st.sidebar.info(f"Failed URLs: {', '.join(failed_urls)}")

    # Add manual documents
    manual_docs = [
        Document(
            page_content="The director of ISET Sfax is not explicitly named in the provided web sources. For the most accurate and current information regarding the director, please contact ISET Sfax directly via their official channels, such as the email address isetsf@isetsf.net or phone numbers listed on their website. As of available but potentially unconfirmed information, Mr. Ahmed Jmal might be associated with the institute.",
            metadata={"source": "manual_director"},
        ),
        Document(
            page_content="ISET Sfax offers national licenses in various fields, including Information Technologies, Mechanical Engineering, Civil Engineering, Process Engineering, and Economics and Management. Specific programs within these departments are detailed on the official ISET Sfax website under the 'Licence Nationale' section. Examples include licenses in Software Engineering, Networks and Services, Mechanical Design, Industrial Maintenance, Civil Construction, Chemical Engineering, Management Informatics, Marketing, Finance, etc.",
            metadata={"source": "manual_licenses"},
        ),
        Document(
            page_content="ISET Sfax has the following main departments: Department of Science Economics and Management, Department of Information Technologies, Department of Process Engineering, Department of Civil Engineering, and Department of Mechanical Engineering. Each department supervises specific national license programs.",
            metadata={"source": "manual_departments"},
        ),
        Document(
            page_content="Contact Email for ISET Sfax: isetsf@isetsf.net. Find phone numbers and the physical address on the official website's contact page.",
            metadata={"source": "manual_contact"},
        ),
    ]
    blog_docs.extend(manual_docs)

    # --- Document Splitting ---
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100  # Adjusted chunking
    )
    splits = text_splitter.split_documents(blog_docs)

    if not splits:
        st.warning(
            "No documents available for RAG. The RAG model will not be functional."
        )

        # Return dummy components that won't work but prevent errors
        class DummyRetriever:
            def get_relevant_documents(self, query):
                return []

        class DummyChain:
            def invoke(self, input):
                return {
                    "answer": "RAG model is unavailable because source documents could not be processed."
                }

        return DummyChain()

    # --- Embedding and Vector Store ---
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

    # Persistent Chroma DB
    persist_directory = "./chroma_db"
    st.sidebar.info(f"Loading/Creating Chroma DB in {persist_directory}...")
    if os.path.exists(persist_directory):
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory, embedding_function=embedding_model
            )
            # Attempt a small query to check if it's functional
            try:
                if vectorstore.get()["ids"]:  # Check if the collection is not empty
                    vectorstore.similarity_search("test query", k=1)
                    st.sidebar.success("Loaded existing Chroma DB.")
                else:
                    raise Exception(
                        "Chroma DB is empty."
                    )  # Treat empty DB as needing rebuild

            except Exception as e:
                st.warning(
                    f"Existing Chroma DB seems corrupted, empty or incompatible. Rebuilding. Error: {e}"
                )
                # Clean up potentially corrupted directory
                if os.path.exists(persist_directory):
                    try:
                        shutil.rmtree(persist_directory)
                    except Exception as clean_e:
                        st.warning(
                            f"Could not remove corrupted DB directory: {clean_e}"
                        )

                # Recreate
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                )
                vectorstore.persist()
                st.sidebar.success("Rebuilt Chroma DB.")

        except Exception as e:
            st.error(f"Fatal error loading Chroma DB: {e}. RAG will be unavailable.")

            # Return dummy chain if DB creation/loading fails critically
            class DummyChain:
                def invoke(self, input):
                    return {
                        "answer": "RAG model is unavailable due to a database error."
                    }

            return DummyChain()

    else:
        st.sidebar.info("Creating new Chroma DB...")
        try:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                persist_directory=persist_directory,
            )
            vectorstore.persist()
            st.sidebar.success("New Chroma DB created.")
        except Exception as e:
            st.error(f"Fatal error creating Chroma DB: {e}. RAG will be unavailable.")

            class DummyChain:
                def invoke(self, input):
                    return {
                        "answer": "RAG model is unavailable due to a database creation error."
                    }

            return DummyChain()

    # --- Retriever and LLM Chain ---
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}  # Increased k for more context
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.1,
        google_api_key=google_api_key,  # Lowered temperature slightly
    )

    # Refined prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant for ISET Sfax.
Answer the user's question truthfully and only based on the following context.
If the context does not contain enough information to answer the question fully or at all, state clearly "I couldn't find the answer to that in my information sources." Do not invent information.
Structure your answer clearly and concisely.
Context: {context}""",
            ),
            ("human", "{input}"),
        ]
    )

    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, Youtube_chain)
    return rag_chain


# Load the RAG pipeline (cached)
rag_chain = load_rag_pipeline(google_api_key)


# Get response based on selected model
def get_response(query, model_choice):
    if model_choice == "NLTK":
        return nltk_model(query)
    elif model_choice == "RAG":
        # Check if rag_chain is functional (not a dummy chain)
        if not hasattr(rag_chain, "invoke"):
            return "The RAG model is not available due to a prior loading error."
        try:
            # Using invoke directly to get the answer part
            response = rag_chain.invoke({"input": query})["answer"]

            # Check if RAG returned its 'couldn't find' fallback
            if (
                "couldn't find the answer" in response.lower()
                or "do not invent information" in response.lower()
            ):
                # You can choose to append a suggestion here or just return the fallback
                return response  # Return the RAG fallback message
            return response  # Return the successful RAG response
        except Exception as e:
            st.error(f"Error during RAG invocation: {e}")
            return "An error occurred while processing your request with the RAG model."
    else:
        return "Invalid model selected."


# Conversation storage (CSV)
def save_conversation_csv(user_id, query, response, model_used):
    conversation_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "query": query,
        "response": response,
        "model_used": model_used,  # Save which model was used
    }
    df = pd.DataFrame([conversation_data])
    chat_history_file = "chat_history.csv"
    try:
        if os.path.exists(chat_history_file) and os.path.getsize(chat_history_file) > 0:
            # Append without header if file exists and is not empty
            df.to_csv(chat_history_file, mode="a", header=False, index=False)
        else:
            # Write with header if file doesn't exist or is empty
            df.to_csv(chat_history_file, index=False)
    except Exception as e:
        st.warning(f"Could not save conversation to '{chat_history_file}': {e}")


# Feedback storage (CSV) - Modified to use message index and filter for NLTK dataset
def save_feedback(message_index, rating, comments):
    if message_index >= len(st.session_state.messages) or message_index < 0:
        st.error("Invalid message index for feedback.")
        return

    msg = st.session_state.messages[message_index]
    query = msg.get("query", "N/A")  # Get query from message metadata
    response = msg.get("content", "N/A")
    model_used = msg.get(
        "model_used", "Unknown"
    )  # Get model used from message metadata

    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "response": response,
        "rating": rating,
        "comments": comments,
        "model_used": model_used,  # Save model used for feedback record
    }
    df = pd.DataFrame([feedback_data])
    feedback_file = "feedback.csv"
    try:
        if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
            df.to_csv(feedback_file, mode="a", header=False, index=False)
        else:
            df.to_csv(feedback_file, index=False)
    except Exception as e:
        st.warning(f"Could not save feedback to '{feedback_file}': {e}")

    # Append to NLTK dataset *only* if thumbs_up AND response was from RAG
    # AND the RAG response wasn't its standard "couldn't find" fallback.
    rag_fallback_keywords = [
        "couldn't find the answer",
        "do not invent information",
    ]  # Keywords to check for RAG fallback
    if (
        rating == "thumbs_up"
        and model_used == "RAG"
        and not any(keyword in response.lower() for keyword in rag_fallback_keywords)
        and query != "N/A"
        and response != "N/A"  # Ensure we have valid Q&A
    ):
        dataset_file = "nltk_dataset.csv"
        dataset_entry = pd.DataFrame([{"question": query, "answer": response}])
        try:
            if os.path.exists(dataset_file) and os.path.getsize(dataset_file) > 0:
                # Check if the Q&A pair might already exist to avoid duplicates? (Optional, complicates code)
                # For simplicity, just append for now.
                dataset_entry.to_csv(dataset_file, mode="a", header=False, index=False)
            else:
                dataset_entry.to_csv(
                    dataset_file, index=False
                )  # Create and write header
            # st.sidebar.info("Added positively rated RAG response to NLTK dataset.") # Can be noisy

        except Exception as e:
            st.warning(f"Could not append to NLTK dataset file '{dataset_file}': {e}")

        # IMPORTANT: New data added to nltk_dataset.csv is NOT automatically reloaded
        # into the running NLTK model (vectorizer/X). It will only be used
        # in future sessions after the app is restarted.


# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("About")
    st.write("This chatbot uses RAG and NLTK to answer questions about ISET Sfax.")
    st.write("You can choose which model to use below.")
    st.write("[Visit ISET Sfax](https://isetsf.rnu.tn/fr)")

    st.session_state.model_choice = st.radio(
        "Choose Model:",
        ("RAG", "NLTK"),
        help="RAG (Retrieval-Augmented Generation) provides detailed answers from web data. NLTK (Natural Language Toolkit) provides simple, pre-defined answers from a CSV dataset.",
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.feedback_submitted = {}  # Clear feedback state too
        st.rerun()

    st.subheader("Persistent Data")
    # Read and display chat history
    chat_history_file = "chat_history.csv"
    if st.button("View Chat History (CSV)"):
        if os.path.exists(chat_history_file):
            try:
                df = pd.read_csv(chat_history_file)
                st.dataframe(
                    df[["timestamp", "user_id", "query", "response", "model_used"]]
                )  # Display model used
            except pd.errors.EmptyDataError:
                st.info("Chat history file is empty.")
            except FileNotFoundError:
                st.info(
                    "No chat history found yet."
                )  # Should be caught by os.path.exists, but good fallback
            except Exception as e:
                st.warning(f"Could not load chat history: {e}")
        else:
            st.info("No chat history found yet.")

    # Read and display feedback
    feedback_file = "feedback.csv"
    if st.button("View Feedback (CSV)"):
        if os.path.exists(feedback_file):
            try:
                df = pd.read_csv(feedback_file)
                st.dataframe(
                    df[["timestamp", "query", "rating", "comments", "model_used"]]
                )  # Display model used
            except pd.errors.EmptyDataError:
                st.info("Feedback file is empty.")
            except FileNotFoundError:
                st.info(
                    "No feedback found yet."
                )  # Should be caught by os.path.exists, but good fallback
            except Exception as e:
                st.warning(f"Could not load feedback: {e}")
        else:
            st.info("No feedback found yet.")

    st.header("Admin Access")
    password = st.text_input("Enter admin password:", type="password")
    if password == "iset_admin_2025":  # Replace with a secure method in production
        st.subheader("Feedback Summary")
        feedback_file = "feedback.csv"
        if os.path.exists(feedback_file):
            try:
                df = pd.read_csv(feedback_file)
                st.write(f"Total feedback entries: {len(df)}")
                if len(df) > 0:
                    positive_count = df["rating"].value_counts().get("thumbs_up", 0)
                    total_feedback = len(df)
                    st.write(
                        f"Positive ratings: {positive_count} ({positive_count/total_feedback:.1%})"
                    )
                    # You could add filtering here, e.g., df[df['model_used'] == 'RAG']
                else:
                    st.write("No feedback yet.")
            except pd.errors.EmptyDataError:
                st.write("Feedback file is empty.")
            except Exception as e:
                st.warning(f"Could not load feedback summary: {e}")
        else:
            st.write("No feedback yet.")

        st.subheader("NLTK Dataset Info")
        dataset_file = "nltk_dataset.csv"
        if os.path.exists(dataset_file):
            try:
                df = pd.read_csv(dataset_file)
                st.write(f"NLTK Q&A pairs in CSV: {len(df)}")
                # st.dataframe(df) # Uncomment to see the raw NLTK dataset
            except pd.errors.EmptyDataError:
                st.write("NLTK dataset file is empty.")
            except Exception as e:
                st.warning(f"Could not load NLTK dataset info: {e}")
        else:
            st.write("NLTK dataset file not found.")

    st.markdown("</div>", unsafe_allow_html=True)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Display content
        st.write(msg["content"])

        # Add feedback buttons only for assistant messages that haven't received feedback
        # Also ensure it's a message we *intended* to rate (i.e., not an error message added manually)
        if (
            msg["role"] == "assistant"
            and idx not in st.session_state.feedback_submitted
            and "query"
            in msg  # Ensure the message has the associated query for feedback
            and msg.get("model_used")
            in ["RAG", "NLTK"]  # Only show feedback for RAG or NLTK responses
        ):
            # Use a unique key for the container to manage state correctly across reruns
            feedback_container_key = f"feedback_container_{idx}"
            # Use a session state variable to track if feedback is being requested for this message (for the text area)
            show_feedback_input = st.session_state.get(
                f"show_feedback_input_{idx}", False
            )

            # Create columns for layout
            col1, col2, col3 = st.columns([1, 1, 8])  # Adjusted columns for layout

            # Display Thumbs Up button
            with col1:
                if st.button("👍", key=f"thumbs_up_{idx}", help="Thumbs Up"):
                    # Call save_feedback with the message index
                    save_feedback(idx, "thumbs_up", "")
                    st.session_state.feedback_submitted[idx] = (
                        "👍"  # Mark as submitted with rating
                    )
                    st.session_state[f"show_feedback_input_{idx}"] = (
                        False  # Hide feedback input if it was open
                    )
                    st.toast("Feedback submitted!")
                    st.rerun()  # Rerun to update UI and hide buttons

            # Display Thumbs Down button
            with col2:
                if st.button("👎", key=f"thumbs_down_{idx}", help="Thumbs Down"):
                    # Toggle the state to show/hide the feedback input for this message
                    st.session_state[f"show_feedback_input_{idx}"] = (
                        not show_feedback_input
                    )
                    # If showing, ensure the text area is visible on rerun
                    st.rerun()

            # Display feedback input if state variable is True
            if show_feedback_input:
                with col3:
                    feedback_text = st.text_area(
                        "Optional comments:", key=f"feedback_text_{idx}", height=50
                    )
                    if st.button("Submit Comments", key=f"submit_feedback_{idx}"):
                        # Call save_feedback with the message index
                        save_feedback(idx, "thumbs_down", feedback_text)
                        st.session_state.feedback_submitted[idx] = (
                            "👎"  # Mark as submitted with rating
                        )
                        st.session_state[f"show_feedback_input_{idx}"] = (
                            False  # Hide input after submission
                        )
                        st.toast("Feedback and comments submitted!")
                        st.rerun()  # Rerun to update UI and hide input

# Close the chat container div
st.markdown("</div>", unsafe_allow_html=True)


# Chat input
query = st.chat_input(
    f"Ask about ISET Sfax using the {st.session_state.model_choice} model:"
)  # Indicate current model
if query:
    # Append user message immediately
    # Store query with user message for feedback context later if needed, and mark it as user
    st.session_state.messages.append(
        {"role": "user", "content": query, "query": query, "model_used": "User"}
    )

    # Before generating response, clear any active feedback input areas
    for idx in range(len(st.session_state.messages)):
        if f"show_feedback_input_{idx}" in st.session_state:
            st.session_state[f"show_feedback_input_{idx}"] = False

    with st.spinner(
        f"Generating response using {st.session_state.model_choice} model..."
    ):
        # Use the selected model from session state
        try:
            response = get_response(query, st.session_state.model_choice)

            # Append assistant message
            # Store query and model used with assistant message for feedback context
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "query": query,
                    "model_used": st.session_state.model_choice,
                }
            )

            # Save conversation to CSV (including model used)
            save_conversation_csv(
                st.session_state.user_id, query, response, st.session_state.model_choice
            )

            # Force rerun to display the new messages and feedback options
            st.rerun()

        except Exception as e:
            # Catch errors during response generation
            error_message = f"Sorry, an error occurred while processing your request with the {st.session_state.model_choice} model: {str(e)}"
            st.error(error_message)
            # Append an error message to history/display
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "query": query,
                    "model_used": "Error",
                }  # Mark as Error
            )
            # Save the conversation, indicating an error occurred
            save_conversation_csv(
                st.session_state.user_id,
                query,
                error_message,
                st.session_state.model_choice + "_Error",
            )
            st.rerun()  # Rerun to show error message
