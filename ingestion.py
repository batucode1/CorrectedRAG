from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
# Yüklenen uzun dokümanları, dil modelinin daha kolay işleyebileceği küçük parçalara ayırmak
# için bir text splitter nesnesi oluşturuyoruz.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
splits = text_splitter.split_documents(docs_list)
# Parçalara ayrılmış metinleri alıp, her birini OpenAIEmbeddings ile vektöre dönüştürerek
# bir Chroma vektör veritabanı oluşturuyoruz.
# Bu işlem, metinlerin anlamsal olarak aranabilir hale gelmesini sağlar
vector_store = Chroma.from_documents(
    documents=splits,
    collection_name="rag",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma"
)
# Diske kaydettiğimiz Chroma veritabanından bir retriever nesnesi oluşturuyoruz.
# Bu nesne, daha sonra bir soru geldiğinde bu veritabanı içinde anlamsal arama yapıp
# soruyla en alakalı metin parçalarını bulmak için kullanılacak.
retriever = Chroma(
    collection_name="rag",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
