import streamlit as st
import subprocess
import sys
import pandas as pd
import numpy as np
import os
import faiss
import base64
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
_env_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_env_dir, ".env"))
load_dotenv(os.path.join(_env_dir, "example.env"))


# Initialize OpenAI client
client = OpenAI(
    api_key= os.getenv('API_KEY'),  # Replace with your actual API key
    base_url="https://integrate.api.nvidia.com/v1"
)

# File paths
metadata_file = "metadata.npy"
index_file = "faiss_index"

# Embedding model (matches the ground-truth notebook) - load once, module level
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Formatting our CSV data
def generate_text_representation(row):
    author_name = row["Author's Name"]
    title = row["Title"]
    publisher = row["Publisher / Place"]
    year = row["Year"]
    return f"{author_name} - {title} - {publisher} - {year}"

# Creating Embeddings and storing it in faiss DB
def create_faiss_database(csv_file):

    df = pd.read_csv(csv_file)
    df["text_representation"] = df.apply(generate_text_representation, axis=1)

    embeddings = embed_model.encode(
        df["text_representation"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    embedding_dim = len(embeddings[0])
    embedding_array = np.array(embeddings).astype('float32')
    embedding_dim = embedding_array.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embedding_array)

    np.save(metadata_file, df["text_representation"].tolist())
    faiss.write_index(index, index_file)
    return index, df["text_representation"].tolist()


# Similarity Search 
def search_faiss(query, top_k=2):
    index = st.session_state.faiss_index
    metadata = st.session_state.faiss_metadata
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True,  # must match indexing
    ).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [
        (metadata[idx], distances[0][i])
        for i, idx in enumerate(indices[0])
        if idx != -1  # faiss returns -1 when fewer than top_k hits
    ]

# Book Description automatic creation
def summarize_book(book_info):
        prompt = f"Provide a concise and clear description for the following book: {book_info}. Avoid introductory phrases like 'Here is a concise description of the book' and focus on the core content."
        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-bolt-e4b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=100,
            stream=False,
        )
        description = response.choices[0].message.content.strip()

        unwanted_phrases = [
            "Here is a concise description of the book:",
            "Here is a brief summary of the book:",
            "This is a brief description of the book:",
            "This book,"
        ]
        for phrase in unwanted_phrases:
            if description.startswith(phrase):
                description = description.replace(phrase, "").strip()

        return description


# Beautify and summarize search results
def beautify_results(results):
    beautified_output = []
    unique_books = {}

    for result, score in results:
        if result not in unique_books:
            unique_books[result] = {"info": result, "distance": score}


    for book in unique_books.values():
        book_info = book["info"]
        description = summarize_book(book_info)
        beautified_output.append({
            "book": book_info.split(' - ')[1],
            "author": book_info.split(' - ')[0],
            "publisher": book_info.split(' - ')[2],
            "year": book_info.split(' - ')[3],
            "description": description
        })

    return beautified_output

# Displaying Result 
def display_results(beautified_results):
    for book in beautified_results:
        st.markdown(f"""
        **Book:** {book['book']}  
        **Author:** {book['author']}  
        **Publisher:** {book['publisher']}  
        **Year:** {book['year']}  
        **Description:** {book['description']}  
        """, unsafe_allow_html=True)


# Data creation for audio
def summarize_results_for_audio(beautified_results):

    output = []
    for idx, book in enumerate(beautified_results, start=1):
        output.append(f"{book['book']} by {book['author']}")

    return "Hello GRIL Trainees, I got these results:\n" + "\n".join(output)


#Generating audio with text
def generate_audio(summary_text):
    command = [
        sys.executable, "python-clients/scripts/tts/talk.py",
        "--server", "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata", "function-id", "877104f7-e885-42b9-8de8-f6e4c6303969",
        "--metadata", "authorization", f"Bearer {os.getenv('API_KEY')}", # Enter your api key
        "--language-code", "en-US",
        "--text", summary_text,
        "--voice", "Magpie-Multilingual.EN-US.Aria",
        "--output", "audio.wav"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Audio generation successful:", result.stdout)
        return "audio.wav"  # Return the audio file path
    except subprocess.CalledProcessError as e:
        print("Error occurred while generating audio:", e.stderr)
        return None
    
# Autoplay audio
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )



# Check for existing FAISS database
def load_faiss_database():
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        loaded_index = faiss.read_index(index_file)
        loaded_metadata = np.load(metadata_file, allow_pickle=True)
        return loaded_index, loaded_metadata
    return None, None

index, metadata = load_faiss_database()

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = index

if "faiss_metadata" not in st.session_state:
    st.session_state.faiss_metadata = metadata

# Audio to text


# Layout Configuration
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 3])

# Left Side: Upload and Create Database
with col1:
    st.header("Upload CSV File")
    
    if index is not None:
        st.success("Database loaded successfully!")
    else:
        st.warning("No Database found. Please upload a CSV file to create one.")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file and st.button("Create Database"):
        with st.spinner("Creating embeddings..."):
            index, metadata = create_faiss_database(uploaded_file)
            st.session_state.faiss_index = index
            st.session_state.faiss_metadata = metadata
            st.success("Database created successfully!")

# Right Side: Query Search
with col2:
    st.image("static/gi_logo.png", width=500)
    st.header("Search Your Book")

    if "recording_status" not in st.session_state:
        st.session_state["recording_status"] = ""
    
    if st.session_state.faiss_index is not None:
        with st.form("search_form"):
            query = st.text_input("Enter query")
            submitted = st.form_submit_button("Search")

        
        if submitted and query.strip() != "":
            results = search_faiss(query)
            beautified_results = beautify_results(results)
            summary_text = summarize_results_for_audio(beautified_results)
            audio_file = generate_audio(summary_text)

            display_results(beautified_results)
            if audio_file:
                autoplay_audio(audio_file)
            else:
                st.warning("Could not generate audio (TTS client unavailable). Results are shown above.")

        # Audio recording section unchanged (currently commented)
        status_placeholder = st.empty()
        
    else:
        st.warning("Please upload a CSV file to create a Database first.")
