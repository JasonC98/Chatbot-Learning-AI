import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from langdetect import detect
import torch

# Global variables to ensure models are loaded only once
MODEL = None
TOKENIZER = None
EMBEDDINGS = None
CHROMA_DB = None
TOEN_TOKENIZER = None
TOEN_MODEL = None

def load_models(model_id, lora_model_path, chroma_dir, embedding_model):
    global MODEL, TOKENIZER, EMBEDDINGS, CHROMA_DB, TOEN_TOKENIZER, TOEN_MODEL

    if MODEL is None:
        TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        base_model = AutoModelForCausalLM.from_pretrained(model_id)
        MODEL = PeftModel.from_pretrained(base_model, lora_model_path)
        MODEL.eval()

    if EMBEDDINGS is None:
        EMBEDDINGS = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})

    if CHROMA_DB is None:
        CHROMA_DB = Chroma(persist_directory=chroma_dir, embedding_function=EMBEDDINGS)

    if TOEN_TOKENIZER is None:
        TOEN_TOKENIZER = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
        TOEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-id-en")

class RAGChatbot:
    def __init__(self, model_id, lora_model_path, chroma_dir, embedding_model):
        load_models(model_id, lora_model_path, chroma_dir, embedding_model)
        self.model = MODEL
        self.tokenizer = TOKENIZER
        self.embeddings = EMBEDDINGS
        self.chroma_db = CHROMA_DB
        self.toEN_tokenizer = TOEN_TOKENIZER
        self.toEN_model = TOEN_MODEL

    def detect_language(self, query):
        try:
            lang = detect(query)
        except:
            lang = "unknown"
        return lang

    def translate_toEN(self, query):
        inputs = self.toEN_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = self.toEN_model.generate(**inputs)
        translated_text = self.toEN_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def retriever(self, query):
        temp_query = query
        if self.detect_language(query) != 'en':
            temp_query = self.translate_toEN(query)

        docs = self.chroma_db.as_retriever().get_relevant_documents(temp_query)
        return docs

    def generate_rag_response(self, query):
        docs = self.retriever(query)

        if not docs:
            return "Sorry, I couldn't find any relevant information." if self.detect_language(query) == 'en' else "Maaf, saya tidak menemukan informasi relevan terkait pertanyaan Anda."

        context = "\n".join([doc.page_content for doc in docs])

        lang = self.detect_language(query)

        past_conversation = "\n".join(st.session_state.chat_memory)

        if lang == 'en':
            prompt = f"""
            Context (in English):
            {context}

            Previous conversation history:
            {past_conversation}

            Question (in English):
            {query}

            Answer in English:
            """
        else:
            prompt = f"""
            Konteks (dalam bahasa Inggris):
            {context}

            Riwayat percakapan sebelumnya:
            {past_conversation}

            Pertanyaan (dalam bahasa Indonesia):
            {query}

            Jawaban dalam bahasa Indonesia:
            """

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if lang == 'en':
            answer = response.split("Answer in English:", 1)[-1].strip()
        else:
            answer = response.split("Jawaban dalam bahasa Indonesia:", 1)[-1].strip()

        st.session_state.chat_memory = [f"User: {query}\nAssistant: {answer}"]

        return answer


# Streamlit app setup
if "chatbot" not in st.session_state:
    model_id = "kalisai/Nusantara-2.7b-Indo-Chat-v0.2"
    lora_model_path = "model_trained_2000/model"
    chroma_dir = "1000_DB"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    st.session_state.chatbot = RAGChatbot(model_id, lora_model_path, chroma_dir, embedding_model)

chatbot = st.session_state.chatbot

st.title("Chatbot to help learn about AI")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello there 👋!\n\nGood to see you, how may I help you today? Feel free to ask me 😁"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = chatbot.generate_rag_response(prompt)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
