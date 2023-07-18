import pickle
import os
import sys

from llama_index import ServiceContext, download_loader, GPTVectorStoreIndex, StorageContext, OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.node_parser.simple import SimpleNodeParser

from langchain.text_splitter import TokenTextSplitter
import tiktoken

import qdrant_client
from qdrant_client.models import Filter, FieldCondition
from qdrant_client.http.exceptions import UnexpectedResponse

import pinecone
from pinecone.core.exceptions import PineconeProtocolError

from elasticsearch import Elasticsearch, NotFoundError

from llama_hub.confluence.base import ConfluenceReader

sys.path.insert(0, './')
import local_secrets as secrets

download_loader("GithubRepositoryReader")
from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader

os.environ['OPENAI_API_KEY'] = secrets.techstyle_openai_key
os.environ['GITHUB_TOKEN'] = secrets.ssk_github_token
os.environ['PINECONE_API_KEY'] = secrets.techstyle_pinecone_api_key

def get_conluence():
    token = {
        'access_token': secrets.techstyle_confluence_token,

    }
    oauth2_dict = {
        "client_id": "SKelley",
        "token": token
    }

    base_url = "https://confluence.techstyle.net/"

    page_ids = ["253301002"]
    space_key = "DataScience"

    reader = ConfluenceReader(base_url=base_url, oauth2=oauth2_dict)
    documents = reader.load_data(space_key=space_key, include_attachments=True, page_status="current")
    documents.extend(reader.load_data(page_ids=page_ids, include_children=True, include_attachments=False))

get_conluence()
exit(0)

def get_repository(owner, repo, branch='main', save_to_disk=True, read_from_disk=True, index_path='./indexes'):
    save_path = f'{index_path}/github_{owner}_{repo}.pkl'
    save_path = save_path.replace(' ', '_')
    docs = None
    if os.path.exists(save_path) and read_from_disk:
        with open(save_path, "rb") as f:
            docs = pickle.load(f)
    else:
        # https://llama-hub-ui.vercel.app/l/github_repo
        github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
        loader = GithubRepositoryReader(
            github_client,
            owner =                  owner,
            repo =                   repo,
            filter_directories =     ([], GithubRepositoryReader.FilterType.EXCLUDE),
            filter_file_extensions = (['.md', '.py', '.ipynb'], GithubRepositoryReader.FilterType.INCLUDE),
            verbose =                False,
            concurrent_requests =    10,
        )
        docs = loader.load_data(branch=branch)
        if save_to_disk:
            with open(save_path, "wb") as f:
                pickle.dump(docs, f)
    for doc in docs:
        doc.extra_info['namespace'] = f'{owner}/{repo}'
        doc.metadata = doc.extra_info
    return docs

def save_doces_to_vector_store(docs, vector_store, chunk_size=1000):
    embed_model = OpenAIEmbedding()
    embed_model.openai_kwargs = {'api_key': secrets.techstyle_openai_key}
    # https://github.com/jerryjliu/llama_index/issues/1206 - handle special tokens like <|endoftext|>
    enc = tiktoken.get_encoding("gpt2")
    tokenizer = lambda text: enc.encode(text, allowed_special={"<|endoftext|>"})
    embed_model._tokenizer = tokenizer
    node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(allowed_special={"<|endoftext|>"}, chunk_size=chunk_size))
    service_context = ServiceContext.from_defaults(embed_model=embed_model, node_parser=node_parser)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    GPTVectorStoreIndex.from_documents(docs, storage_context=storage_context, service_context=service_context)

def save_docs_to_qdrant(docs, index_name, namespace, chunk_size=1000, vector_length=1536):
    client = qdrant_client.QdrantClient(host='localhost', port=6333)
    query_filter = query_filter=Filter(must=[FieldCondition(key='namespace', match={'value': f'{namespace}'})])
    try:
        client.delete(collection_name='github', points_selector=query_filter)
    except UnexpectedResponse:
        pass
    vector_store=QdrantVectorStore(client=client, collection_name=index_name, namespace=namespace)
    save_doces_to_vector_store(docs, vector_store, chunk_size) 

def save_docs_to_pinecone(docs, index_name, namespace, chunk_size=1000, vector_length=1536):
    client = pinecone.init(api_key=secrets.techstyle_pinecone_api_key, environment='us-east-1-aws')
    try:
        pinecone.Index(index_name).delete(deleteAll='true', namespace=namespace)
    except PineconeProtocolError:
        pinecone.create_index(index_name, vector_length)
    vector_store=PineconeVectorStore(client=client, index_name=index_name, namespace=namespace, environment='us-east-1-aws')
    save_doces_to_vector_store(docs, vector_store, chunk_size) 

def save_docs_to_elasticsearch(docs, index_name, namespace, chunk_size=1000, vector_length=1536):
    client = OpensearchVectorClient(endpoint='http://localhost:9200', index=index_name, dim=vector_length)
    es_client = Elasticsearch(hosts=["http://localhost:9200"], verify_certs=False)
    try:
        query = {"bool" : {"must" : [{"term" : {"metadata.namespace.keyword" : namespace }}]}}
        es_client.delete_by_query(index='github', query=query)
    except NotFoundError:
        pass
    vector_store=OpensearchVectorStore(client=client)
    save_doces_to_vector_store(docs, vector_store, chunk_size) 


index = 'github'
owner = 'stephenskelley'
repo = 'langchain_play'
namespace = f'{owner}/{repo}'
docs = get_repository(owner=owner, repo=repo)
save_docs_to_qdrant(docs, index, namespace)
save_docs_to_pinecone(docs, index, namespace)
save_docs_to_elasticsearch(docs, index, namespace)
exit(0)
