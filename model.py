import pickle as pkl
import os
import toml

import openai
# from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
# from langchain.agents import initialize_agent, Tool
# from langchain.agents import AgentType
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
import aiohttp
import asyncio
from langchain.vectorstores.faiss import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings

from postprocess import parse_url, parse_element

########### ===== config ===== #############
config_path = '.streamlit/secrets.toml'
if os.path.exists(config_path):
	print(f"{config_path} exists")
	config = toml.load(open( config_path, 'r'))
else:
	print( f"secrets -> {st.secrets}" )
	config = dict(st.secrets.items())
print( f"config -> {config}")
openai.api_key = os.environ.get("OPENAI_API_KEY") or config["settings"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai.api_key
########### ==================== #############


def get_chain():
	qa_template = '''
	As a doctor's assistant, you will be provided with a document delimited by triple quotes and a question. 
	Your task is to answer the question in traditional Chinese using only the provided document and to extract the source(s) and passage(s) in the document used to answer the question. 
	The answer must be helpful and self-explanatory. Since you are a doctor's assistant, you are expected to answer the question in a professional manner.
	Always use adverbs like probably, likely and possibly to avoid giving medical advice. Remind the patient to consult a doctor or visit a clinic if necessary.
	If the document does not contain information needed to answer this question then simply summarize the provided documents like "沒有從部落格中找到合適回答的內容，但找到以下資訊：..." 
	If an answer to the question is provided, the source(s) must be annotated as complete URL(s) at the end of the answer, and its format is as follows: 【``Source``: ......】.


	Document: """{context}"""


	Question: {question}


	Answer:'''

	qa_prompt = PromptTemplate(template = qa_template, input_variables=["context", "question"])
	qa_llm = OpenAI(temperature=0, model = "gpt-3.5-turbo-instruct", max_tokens = 1536)

	chain = load_qa_with_sources_chain( qa_llm, prompt = qa_prompt, document_variable_name = "context", chain_type="stuff")

	return chain


def get_vectorstore(file_path: str):
	"""
	"""
	with open( file_path, "rb") as f:
		vectorstore = pkl.load(f)
	return vectorstore


def get_response( chain, vectorstore, question: str, topn: int):
	"""
	Argument
	--------
	chain: langchain.chains.qa_with_sources.QAWithSourcesChain

	vectorstore: langchain.vectorstore.VectorStore

	question: str
		e.g. "肩膀不太舒服，前幾天有去打球，會不會是韌帶拉傷？"

	"""
	rele_docs = vectorstore.similarity_search(query = question, k = topn)
	res = chain({"input_documents": rele_docs, "question": question}, return_only_outputs=False)
	print( f"search res -> {res}" )
	return res


def get_answer( question: str) -> dict:
	"""
	Argument
		question: str
	"""
	res = get_response( chain, vectorstore, question, topn=2)
	try:
		res["output_text"] = res["output_text"].replace("】", " 】").replace("【", "\n【 ")
		res["input_documents"] = [ doc.page_content for doc in res["input_documents"]]
	except Exception as e:
		print( f"#ERROR: e is {e}, res is {res}")

	try:
		# urls = parse_url( res["output_text"] )
		# res["urls"] = urls
		rev_check_docs = vectorstore.similarity_search(query = res["output_text"], k = 2)
		res["urls"] = [rev_check_docs[0].metadata['url']]
	except Exception as e:
		print( f"#ERROR: e is {e}, res is {res}")
	print(f"parsed res -> {res}")
	return res


async def aget_answer(question):
	async with aiohttp.ClientSession() as session:
		async with session.post('http://127.0.0.1:9001/webqa', json = {'question': question}) as response:
			print( f"Status: {response.status}")
			print( f"Content-type: {response.headers['content-type']}")
			content = await response.json()
			print( f"Body: {content}")
	return content

print("Loading chain...")
chain = get_chain()
print("Loading vectorstore...")
# vectorstore = get_vectorstore("data/pain/vectorstore/vectorstore.pkl")
vectorstore = FAISS.load_local("data/pain/vectorstore/vectorstore.pkl", OpenAIEmbeddings(), allow_dangerous_deserialization = True )
