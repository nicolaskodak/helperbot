import pickle as pkl

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
import aiohttp
import asyncio

from postprocess import parse_url, parse_element


def get_chain():
	qa_template = '''
	You will be provided with a document delimited by triple quotes and a question. 
	Your task is to answer the question in traditional Chinese using only the provided document and to extract the source(s) and passage(s) in the document used to answer the question. 
	If the document does not contain information needed to answer this question then simply summarize the provided documents like "沒有找到最直接的答案，但找到以下資訊：..." 
	If an answer to the question is provided, the source(s) must be annotated as complete URL(s) at the end of the answer, and its format is as follows: 【``Source``: ......】.


	Document: """{context}"""


	Question: {question}


	Answer:'''

	qa_prompt = PromptTemplate(template = qa_template, input_variables=["context", "question"])
	qa_llm = OpenAI(temperature=0, max_tokens = 1024)

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
	# print( res )
	return res


def get_answer( question: str) -> dict:
	"""
	Argument
		question: str
	"""
	res = get_response( chain, vectorstore, question, topn=2)
	res["output_text"] = res["output_text"].replace("】", " 】").replace("【", "\n【 ")
	
	urls = parse_url( res["output_text"] )
	res["urls"] = urls
	return res


async def aget_answer(question):
	async with aiohttp.ClientSession() as session:
		async with session.post('http://127.0.0.1:9001/webqa', json = {'question': question}) as response:
			print("Status:", response.status)
			print("Content-type:", response.headers['content-type'])
			content = await response.json()
			print("Body:", content, "...")
	return content

print("Loading chain...")
chain = get_chain()
print("Loading vectorstore...")
vectorstore = get_vectorstore("data/pain/vectorstore/vectorstore.pkl")
