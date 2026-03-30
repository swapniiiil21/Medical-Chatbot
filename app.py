from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    file = request.files.get("file")
    
    if file and file.filename != '':
        import tempfile
        import textwrap
        from langchain_community.document_loaders import PyPDFLoader
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            try:
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                pdf_text = " ".join([d.page_content for d in docs])
                pdf_text = textwrap.shorten(pdf_text, width=15000, placeholder="... [Document Trimmed]")
                
                custom_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt + "\n\n[USER REFERENCE DOCUMENT]\n{context}\n\nUse this document to help answer the user's question."),
                    ("human", "{question}")
                ])
                custom_chain = custom_prompt | chatModel
                response = custom_chain.invoke({"context": pdf_text, "question": msg})
                ans = response.content
            except Exception as e:
                ans = f"Error processing uploaded document: {str(e)}"
            finally:
                os.remove(tmp.name)
    else:
        # Standard Pinecone QA
        response = rag_chain.invoke({"input": msg})
        ans = response["answer"]
        
    print("Response : ", ans)
    return str(ans)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
