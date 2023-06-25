import os
import time
import pickle as pkl
import re
import yaml
import toml

import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

import openai

from dotenv import load_dotenv
load_dotenv()

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
	print( res )
	return res


chain = get_chain()
vectorstore = get_vectorstore("data/pain/vectorstore/vectorstore.pkl")

import re
url_pattern = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

def parse_url( answer: str) -> list:
	"""
	Argument
		--------
		answer: str
			e.g. "【  `Source`: https://draiden.org/acromioclavicular-joint-injury/  】"
	"""
	return [ x[0].replace("】","").replace("【", "") for x in url_pattern.findall(answer)]


def parse_element(answer: str, element: str):
	"""
	Arguments:
	"""
	sq_bracket_pattern = re.compile("\【[[^\【^\】^`]*``{element}``:([^\【^\】^`]+)\】")
	found = sq_bracket_pattern.findall(answer)
	parsed = None
	if found and len(found)>0:
		parsed = found[0][0].strip()
	return { element: parsed}

########### ===== main app ===== #############

### ===== Creating a login widget ===== ###
config_path = '.streamlit/config.toml'
if os.path.exists(config_path):
	config = toml.load(open( config_path, 'r'))
	authenticator = stauth.Authenticate(
		config['credentials'],
		config['cookie']['name'],
		config['cookie']['key'],
		config['cookie']['expiry_days'],
		config['preauthorized']
	)
else:
	authenticator = stauth.Authenticate(
		st.secrets['credentials'],
		st.secrets['cookie']['name'],
		st.secrets['cookie']['key'],
		st.secrets['cookie']['expiry_days'],
		st.secrets['preauthorized']
	)


name, authentication_status, username = authenticator.login('Login', 'main')


### ===== Authenticating users ===== ###
if authentication_status:
	col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)
	with col4:
		# st.write(f'Welcome *{name}*')
		st.write(f'Welcome!')
		# st.write(f'*{name}*')
		# st.markdown(f"<p style='text-align: center;'> {name} </p>", unsafe_allow_html=True)
		authenticator.logout('Logout', 'main')
	# st.write(f'\n\n\n\n')
	
	with st.sidebar:
		# serper_api_key = st.text_input('Serper API Key',key='langchain_search_api_key_serper')
		# openai_api_key = st.text_input('OpenAI API Key',key='langchain_search_api_key_openai')
		openai.api_key = os.environ.get("OPENAI_API_KEY")

	st.title("生產力小幫手")
	# st.divider() 


	tab1, tab2 = st.tabs(["回覆小幫手", "等待更多點子"])

	with tab1:
		question = st.text_input("病患描述問題", placeholder="肩膀不太舒服，前幾天有去打球，會不會是韌帶拉傷？")

		if question:
			res = get_response( chain, vectorstore, question, topn=2)
			res["output_text"] = res["output_text"].replace("】", " 】").replace("【", "\n【 ")
			st.write("Answer")
			st.write(res["output_text"])
			# parsed = parse_element( res["output_text"], element = "Source"  )
			# print( parsed )
			urls = parse_url( res["output_text"] )
			print( f"found urls -> {urls}" )
			html = ""
			if len(urls)>0:
				for url in urls:
					html += f"""<br><a href=\"{url}\">開啟網頁：{url}</a>"""
			html = f"<div>{html}</div>"
			components.html( html, height=600, scrolling=True)


elif authentication_status == False:
	st.error('Username/password is incorrect')
	
elif authentication_status == None:
	st.warning('Please enter your username and password')

