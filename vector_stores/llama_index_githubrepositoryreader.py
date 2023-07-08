# https://github.com/jerryjliu/llama_index/issues/1194
# note this GithubRepositoryReader is different than https://llama-hub-ui.vercel.app/l/github_repo
import os
import sys
from llama_index import GithubRepositoryReader, GPTVectorStoreIndex
sys.path.insert(0, './')
import local_secrets as secrets

os.environ['GITHUB_TOKEN'] = secrets.ssk_github_token
loader = GithubRepositoryReader(
    github_token=os.environ["GITHUB_TOKEN"],
    owner="jerryjliu",
    repo="llama_index",
    use_parser=False,
    verbose=True,
    ignore_directories=["examples"],
)
docs = loader.load_data(branch='main')
for doc in docs:
    print(doc.extra_info)
index = GPTVectorStoreIndex.from_documents(docs)
index.save_to_disk('data/llama_index.index.json')