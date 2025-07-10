import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from langsmith import Client
def get_graph():
    # Initialize the LLM
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMENI_API_KEY")
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    # Initialize your embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.load_local(
        folder_path="/content/drive/My Drive/10Academy/Week6/Data/faiss_with_metadata/",
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )

    client = Client(api_key=LANGSMITH_API_KEY)
    prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str


    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=5)
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}


    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": "What is the main complaints customers have?"})
    return result

graph = get_graph()

result = graph.invoke({"question": "What is the main complaints customers have?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')