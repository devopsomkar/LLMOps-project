from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# Removed ChatOpenAI import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
load_dotenv()

# -----------------------------
# Pinecone API Key setup
# -----------------------------
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# -----------------------------
# Hugging Face embeddings
# -----------------------------
embeddings = download_hugging_face_embeddings()

# -----------------------------
# Pinecone Vector Store
# -----------------------------
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# -----------------------------
# Hugging Face model setup
# -----------------------------
hf_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-large",  # or use flan-t5-base/small for faster practice
    tokenizer="google/flan-t5-large",
    max_length=512
)
chatModel = HuggingFacePipeline(pipeline=hf_pipeline)

# -----------------------------
# Prompt & Chain setup
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("Input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

# -----------------------------
# Run the app
# -----------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
