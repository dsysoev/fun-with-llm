import argparse

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to PDF document")
    parser.add_argument(
        "--model",
        default="llama2",
        type=str,
        help="Ollama model name (see details https://ollama.com/library)",
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:11434",
        type=str,
        help="""Base url the model is hosted under.""",
    )
    parser.add_argument(
        "--temperature",
        default=0.8,
        type=float,
        help="""The temperature of the model. Increasing the temperature will
            make the model answer more creatively. (Default: 0.8)""",
    )

    args = parser.parse_args()

    # load model for prompt and prepare embeddings
    llm = Ollama(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
    )
    embeddings = OllamaEmbeddings(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
    )
    # for split pages
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    # read pdf document
    loader = PyPDFLoader(args.path)
    pages = loader.load_and_split()
    all_splits = text_splitter.split_documents(pages)
    # create embeddings based for each chunk of document
    # see details
    # https://python.langchain.com/docs/modules/data_connection/vectorstores/
    # https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md
    store = Chroma.from_documents(all_splits, embeddings)
    qachain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

    while True:
        print("#" * 23)
        prompt = input("Input your prompt here: ")
        print("#" * 23)
        if prompt == "/bye":
            break
        answer = qachain.invoke({"query": prompt})
        print(answer["result"])


if __name__ == "__main__":
    main()
