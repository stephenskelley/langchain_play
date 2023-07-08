import pickle
import os
from llama_index import ServiceContext, download_loader, GPTVectorStoreIndex, StorageContext, OpenAIEmbedding
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain.vectorstores import Qdrant
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.node_parser.simple import SimpleNodeParser
from langchain.text_splitter import TokenTextSplitter
import tiktoken
from langchain.text_splitter import CharacterTextSplitter
import qdrant_client
import pinecone
import sys
sys.path.insert(0, './')
import local_secrets as secrets
download_loader("GithubRepositoryReader")
from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader
os.environ['OPENAI_API_KEY'] = secrets.techstyle_openai_key
os.environ['GITHUB_TOKEN'] = secrets.ssk_github_token
os.environ['PINECONE_API_KEY'] = secrets.techstyle_pinecone_api_key

def get_repository(owner, repo, branch='main', save_to_disk=True, read_from_disk=True, index_path='./indexes'):
    save_path = f'{index_path}/github_{owner}_{repo}.pkl'
    save_path = save_path.replace(' ', '_')
    docs = None
    if os.path.exists(save_path) and read_from_disk:
        with open(save_path, "rb") as f:
            docs = pickle.load(f)
    else:
        github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
        loader = GithubRepositoryReader(
            github_client,
            owner =                  owner,
            repo =                   repo,
            filter_directories =     (['docs', 'llama_index', 'examples'], GithubRepositoryReader.FilterType.INCLUDE),
            filter_file_extensions = (['.md', '.py', '.ipynb'], GithubRepositoryReader.FilterType.INCLUDE),
            # verbose =                True,
            concurrent_requests =    10,
        )
        docs = loader.load_data(branch=branch)
        if save_to_disk:
            with open(save_path, "wb") as f:
                pickle.dump(docs, f)
    for doc in docs:
        doc.extra_info['repository'] = '{owner}/{repo}'
        doc.metadata = doc.extra_info
    return docs

def save_docs_to_qdrant(docs, index_name, namespace):
    break


docs = get_repository(owner='jerryjliu', repo='llama_index')
save_docs_to_qdrant(docs, 'github', 'jerryjliu/llama_index')

exit(1)

embed_model = OpenAIEmbedding()
embed_model.openai_kwargs = {'api_key': secrets.techstyle_openai_key}

# https://github.com/jerryjliu/llama_index/issues/1206 - handle special tokens like <|endoftext|>
node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(allowed_special={"<|endoftext|>"}))
service_context = ServiceContext.from_defaults(embed_model=embed_model, node_parser=node_parser)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter=TokenTextSplitter(allowed_special={"<|endoftext|>"})
docs = text_splitter.split_documents(docs)
qdrant = Qdrant.from_documents(docs, OpenAIEmbeddings(), url='http://localhost:6333', collection_name="github")



# qdrant
#client = qdrant_client.QdrantClient(url='http://localhost:6333')
#vector_store = QdrantVectorStore(client=client, collection_name='github', namespace='llama_index')

#pinecone
#client = pinecone.init(api_key=secrets.techstyle_pinecone_api_key, environment='us-east-1-aws')
#vector_store=PineconeVectorStore(client=client, index_name='ssk', namespace='github-llama-index', environment='us-east-1-aws')

#storage_context = StorageContext.from_defaults(vector_store=vector_store)
#index = GPTVectorStoreIndex.from_documents(docs, storage_context=storage_context, service_context=service_context)
