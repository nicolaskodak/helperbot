import os
import time
import pickle as pkl
import re
import yaml
import toml

import aiohttp
import asyncio
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

from model import get_answer
from postprocess import parse_url, parse_element

########### ===== main app ===== #############
async def aget_answer(question):
	async with aiohttp.ClientSession() as session:
		async with session.post('http://127.0.0.1:9001/webqa', json = {'question': question}) as response:
			print("Status:", response.status)
			print("Content-type:", response.headers['content-type'])
			content = await response.json()
			print("Body:", content, "...")
	return content

### ===== Creating a login widget ===== ###
config_path = '.streamlit/secrets.toml'
if os.path.exists(config_path):
	print(f"{config_path} exists")
	config = toml.load(open( config_path, 'r'))
else:
	print( f"secrets -> {st.secrets}" )
	config = dict(st.secrets.items())
print( f"config -> {config}")

authenticator = stauth.Authenticate(
		config['credentials'],
		config['cookie']['name'],
		config['cookie']['key'],
		config['cookie']['expiry_days'],
		config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')
is_production = os.environ.get("PRODUCTION") or config["PRODUCTION"]
print(f"is_production -> {is_production}")

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
		openai.api_key = os.environ.get("OPENAI_API_KEY") or config["OPENAI_API_KEY"]

	st.title("生產力小幫手")
	# st.divider() 


	tab1, tab2 = st.tabs(["回覆小幫手", "等待更多點子"])

	with tab1:
		question = st.text_input("病患描述問題", placeholder="肩膀不太舒服，前幾天有去打球，會不會是韌帶拉傷？")

		if question:
			if is_production:
				res = get_answer(question)
			else:
				res = asyncio.run(aget_answer(question))

			urls = res['urls']

			st.write("建議回覆")
			st.write(res["output_text"])
			

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

