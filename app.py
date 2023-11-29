import os
import time
import pickle as pkl
# import re
# import yaml
import toml
import logging
from datetime import date

# import aiohttp
import pandas as pd
from pytrends.request import TrendReq
import serpapi
import asyncio
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import langchain
# from langchain.utilities import GoogleSerperAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
import openai

from dotenv import load_dotenv
load_dotenv()


########### ===== config ===== #############
config_path = '.streamlit/secrets.toml'
# print(f" os.listdir('.') -> { os.listdir('.')}")

if os.path.exists(config_path):
	logging.info(f"{config_path} exists")
	config = toml.load(open( config_path, 'r'))
else:
	logging.info( f"secrets -> {st.secrets}" )
	config = dict(st.secrets.items())
print( f"config -> {config}")
for k in ['name', 'authentication_status', 'username' ]:
	st.session_state[k] = None
logging.info(f"session state -> {st.session_state}")
openai.api_key = os.environ.get("OPENAI_API_KEY") or config["settings"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai.api_key


secret_key = os.environ.get('SERP_APIKEY') or config["settings"]["SERP_APIKEY"]
logging.info(f"serp api key -> {secret_key}")
# serp_client = serpapi.Client(api_key=secret_key)
pytrends = TrendReq(hl='zh-TW', tz=480, timeout=(10,25), retries=2, backoff_factor=0.1, requests_args={'verify':False})

########### ==================== #############


### ===== Creating a login widget ===== ###
authenticator = stauth.Authenticate(
		config['credentials'],
		config['cookie']['name'],
		config['cookie']['key'],
		config['cookie']['expiry_days'],
		config['preauthorized']
)
name, authentication_status, username = authenticator.login('Login', 'main')
is_production = os.environ.get("PRODUCTION") or config["settings"]["PRODUCTION"]
logging.info(f"is_production -> {is_production}")
########### ==================== #############

secret_key = os.environ.get('SERP_APIKEY')
pytrends = TrendReq(hl='zh-TW', tz=480, timeout=(10,25), retries=2, backoff_factor=0.1, requests_args={'verify':False})

@st.cache_data
def get_trending_searches(today) -> pd.DataFrame:
    """
    Get trending searches from Google Trends
    """
    results = pytrends.trending_searches(pn='taiwan')
    return results

@st.cache_data
def get_search_results( secret_key: str, query: str, today: str) -> dict:
    """
    Get search results from serpapi.com
    """
    print(f"secret_key -> {secret_key[-4:]}, query -> {query}, today -> {today}")
    serp_client=serpapi.Client(api_key=secret_key)
    results = serp_client.search({
        'engine': 'google',
        'q': query,
    })
    return results.as_dict()


audio_input_dir = "data/audio"
audio_output_dir = "data/audio"
transcript_input_dir = "data/transcripts"
transcript_output_dir = "data/transcripts"
summary_input_dir = "data/summary"
summary_output_dir = "data/summary"

### === load model & utilities === ###
from model import get_answer, aget_answer
from postprocess import parse_url, parse_element
from utils import split, compose_file_path, get_rewrite_chain, get_summarize_chain, sound2txt, text2summary	

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

	st.title("生產力小幫手")
	# st.divider() 


	tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["疼痛小幫手", "切割錄音檔", "錄音轉文字", "文字改寫", "總結摘要", "錄音檔摘要", "火熱話題"])

	with tab1:
		question = st.text_input("病患描述問題", placeholder="肩膀不太舒服，前幾天有去打球，會不會是韌帶拉傷？")

		if question:
			if is_production:
				res = get_answer(question)
			else:
				res = asyncio.run(aget_answer(question))

			urls = res['urls']

			st.write("【建議回覆】")
			st.write(res["output_text"])
			

			print( f"found urls -> {urls}" )
			html = ""
			if len(urls)>0:
				for url in urls:
					html += f"""<br><a href=\"{url}\" style="color:gray">開啟網頁：{url}</a>"""
			html = f"<div>{html}</div>"
			components.html( html, height=600, scrolling=True)

	with tab2: 
		## --- upload file --- ##
		with st.form("upload-audio-form", clear_on_submit=True):
			uploaded_file = st.file_uploader("Upload an audio clip", key='split_audio', type=["wav", "mp3", "m4a"])
			submitted = st.form_submit_button("UPLOAD")
			if submitted and uploaded_file:
				## -- read file -- ##
				sound_bytes = uploaded_file.getvalue() # bytes
				
				file_obj = compose_file_path( uploaded_file.name, output_prefix = None, output_dir = audio_input_dir, output_ext = None)
				audio_file_path = file_obj['output_file_path']
				print( f"audio_file_path: { audio_file_path}")
				with open( audio_file_path, "wb") as f:
					f.write(sound_bytes)

				## --- split --- ##
				with st.spinner(f"reading from { audio_file_path}"):
					multi_file_paths = split( in_file_path = audio_file_path, length = 10)

	with tab3: 
		## --- upload file --- ##
		transcript_file_path = ""
		transcripted = 0
		with st.form("upload-audio-clip-form", clear_on_submit=True):
			uploaded_file = st.file_uploader("Upload an audio clip", key = 'transcribe_audio', type=["wav", "mp3", "m4a"])
			submitted = st.form_submit_button("UPLOAD")
			if submitted and uploaded_file:
				## -- read file -- ##
				sound_bytes = uploaded_file.getvalue() # bytes
				file_obj = compose_file_path( input_file_path=uploaded_file.name, output_prefix = None, output_dir = audio_input_dir, output_ext = None)
				audio_file_path = file_obj['output_file_path']
				print( f"audio_file_path: { audio_file_path}")
				with open( audio_file_path, "wb") as f:
					f.write(sound_bytes)

				## --- transcribe --- ##		
				start_time = time.time()
				st.divider()	
				file_obj = compose_file_path( audio_file_path, output_prefix = None, output_dir = transcript_input_dir, output_ext = "txt")
				transcript_file_path = file_obj['output_file_path']
				if os.path.exists(transcript_file_path): 
					print(f"{transcript_file_path} already exists. will skip this file.")	 
				with st.spinner(f"written transcript to { transcript_file_path}"):
					transcript = sound2txt( audio_file_path, transcript_file_path)
					st.write(transcript['text'])
				print( f"Elapsed time: {time.time() - start_time} seconds")
			transcripted = 1
		if transcripted and os.path.exists(transcript_file_path):
			with open(transcript_file_path) as f:
				st.download_button('下載逐字稿', f, file_name = os.path.basename(transcript_file_path), )  # Defaults to 'text/plain'

	with tab4:
		
		rewrite_chain = get_rewrite_chain()
		rewrite_file_path = ""
		rewritten = 0
		question = st.text_input(
				"你希望整理會議逐字稿的方式（用簡短的英文描述）",
				"correct typos, remove redundant sentences/words and add punctuations",
				placeholder="correct typos, remove redundant sentences/words and add punctuations",
				# disabled=not uploaded_file,
			)
		
		with st.form("upload-transcript-clip-form", clear_on_submit=True):
			rewrite_uploaded_file = st.file_uploader("Upload a transcript or a document (txt file)", type="txt")
			submitted = st.form_submit_button("UPLOAD")
			
			if submitted and rewrite_uploaded_file and question and not openai.api_key:
				st.info("Please add your OPENAI API KEY to environment to continue.")
				
			if submitted and rewrite_uploaded_file and question and openai.api_key:
				## -- read file -- ##
				article = rewrite_uploaded_file.read().decode()
				
				file_obj = compose_file_path( 
					input_file_path = rewrite_uploaded_file.name, output_prefix = "rewrite", output_dir = transcript_output_dir, output_ext = "txt"
				)
				rewrite_file_path = file_obj['output_file_path']
				# assert not os.path.exists( transcript_file_path), f"File { transcript_file_path} already exists."

				## -- split into chunks -- ##
				text_splitter = RecursiveCharacterTextSplitter(
					chunk_size=1024,
					chunk_overlap=20,
				)
				paragraphs = text_splitter.split_text(article)

				## --- anthropic --- ##
				# prompt = f"""
				# {anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
				# {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}
				# """
				# client = anthropic.Client(anthropic_api_key)
				# #client = anthropic.Client(st.secrets.anthropic_api_key)
				# response = client.completion(
				#     prompt=prompt,
				#     stop_sequences=[anthropic.HUMAN_PROMPT],
				#     model="claude-v1",
				#     max_tokens_to_sample=100,
				# )
				# st.write("### Answer")
				# st.write(response["completion"])

				## --- openai --- ##
				
				st.write("### Rewritten transcript")
				with st.spinner(f"written to { rewrite_file_path}"):
					for paragraph in paragraphs: 
						try:
							res = rewrite_chain({"functions": question, "paragraph": paragraph}, return_only_outputs = True)
						except Exception as e:
							print(e)
							continue
						st.write(res["text"])
						st.write("---")
						with open( rewrite_file_path, "a") as f:
							f.write(res["text"])
						# rewrite_file_paths.append( rewrite_file_path)
			rewritten = 1

		if rewritten and os.path.exists(rewrite_file_path):
			with open(rewrite_file_path) as f:
				st.download_button('下載加入標點符號後的逐字稿', f, file_name = os.path.basename(rewrite_file_path),  key = f"download-button-{rewrite_file_path}")  # Defaults to 'text/plain'

	with tab5: 
		instruction = st.text_input(
				"你希望總結會議逐字稿的方式（用簡短的英文描述）",
				"give helpful and concise summary with action items in traditional chinese.",
				placeholder="give helpful and concise summary with action items in traditional chinese.",
				# disabled=not summary_uploaded_file,
			)

		summarized = 0
		summary_file_path = ""
		with st.form("upload-summarize-clip-form", clear_on_submit=True):
			summary_uploaded_file = st.file_uploader("Upload a transcript or a document (txt file)", type="txt")
			summarize_submitted = st.form_submit_button("UPLOAD")
			
			if summarize_submitted and summary_uploaded_file and instruction and not openai.api_key:
				st.info("Please add your OPENAI API KEY to environment to continue.")
				
			if summarize_submitted and summary_uploaded_file and instruction and openai.api_key:
				## -- read file -- ##
				article = summary_uploaded_file.read().decode()
				
				file_obj = compose_file_path( 
					input_file_path = summary_uploaded_file.name, output_prefix = "summary", output_dir = summary_output_dir, output_ext = "txt"
				)
				summary_file_path = file_obj['output_file_path']

				## -- split into chunks -- ##
				text_splitter = RecursiveCharacterTextSplitter(
					chunk_size=1024,
					chunk_overlap=100,
				)
				paragraphs = text_splitter.split_text(article)

				## -- summarize -- ##
				st.write("### Summarize transcript")
				with st.spinner(f"written to { summary_file_path}"):
					summarize_chain = get_summarize_chain()
					with open( summary_file_path, "w") as f:
						for paragraph in paragraphs: 
							try:
								res = summarize_chain( {"input_documents": [ Document( page_content=paragraph) ], "functions": instruction} )
								summary = res['output_text']
							except Exception as e:
								print(e)
								continue
							st.write(summary)
							st.write("---")
							f.write(summary)
			summarized = 1

		if summarized and os.path.exists(summary_file_path):
			with open(summary_file_path) as f:
				st.download_button('下載摘要', f, file_name = os.path.basename(summary_file_path), key = f"download-button-{summary_file_path}")  # Defaults to 'text/plain'

	with tab6: 
		# summarize_chain = load_summarize_chain()
		# data/audio/GMT20230613-051643_Recording.m4a_0.m4a
		# with open("data/transcripts/GMT20230613-051643_Recording.m4a_0.txt") as f:
		# 	transcript = f.read()
		# st.markdown( f'<p style="background-color: rgba(100,100,200,.2);">{transcript}</p>', unsafe_allow_html=True)

		## --- upload file --- ##
		with st.form("end-to-end-upload-audio-form", clear_on_submit=True):
			uploaded_file = st.file_uploader("Upload an audio clip", key='end_to_end', type=["wav", "mp3", "m4a"])
			submitted = st.form_submit_button("UPLOAD")
			if submitted and uploaded_file:
				## -- read file -- ##
				with st.spinner(f"uploaded { uploaded_file.name}"):
					sound_bytes = uploaded_file.getvalue() # bytes
					audio_file_path = compose_file_path( input_file_path = uploaded_file.name, output_prefix = None, output_dir = audio_input_dir, output_ext = None)['output_file_path']
					print( f"audio_file_path: {audio_file_path}")
					with open( audio_file_path, "wb") as f:
						f.write(sound_bytes)
					## --- split --- ##		
					audio_file_paths = split( in_file_path = audio_file_path, length = 10)

				## --- transcribe --- ##		
				st.write("### Transcribing audio")
				with st.spinner(f"transcribe { ', '.join(audio_file_paths) }"):
					transcripts = []
					transcript_file_paths = []
					for i, audio_file_path in enumerate(audio_file_paths):
						start_time = time.time()
						
						transcript_file_path = compose_file_path( input_file_path = audio_file_path, output_prefix = None, output_dir = transcript_input_dir, output_ext = "txt")['output_file_path']
						transcript_file_paths.append(transcript_file_path)

						st.write(f"Trascribing {i}-th clip out of total {len(audio_file_paths)} clips.")
						transcript = sound2txt( audio_file_path, transcript_file_path)

						st.write(transcript['text'][:50])
						st.divider()
						transcripts.append( transcript['text'])
						print( f"[transcription] Elapsed time: {time.time() - start_time} seconds")
					# print( f"Transcript: {transcript}")
				
				## --- Rewrite --- ##
				st.write("### Rewritten transcript")
				with st.spinner(f"rewritten { ','.join(transcript_file_paths)}"):
					text_splitter = RecursiveCharacterTextSplitter(
						chunk_size=1024,
						chunk_overlap=20,
					)
					rewrite_instruction = "correct typos, remove redundant sentences/words and add punctuations"
					rewrite_file_paths = []
					rewrites = []
					for i, blob in enumerate(zip(transcript_file_paths, transcripts)):
						transcript_file_path, transcript = blob
						paragraphs = text_splitter.split_text( "\n".join(transcript) )
						rewrite_file_path = compose_file_path( input_file_path = transcript_file_path, output_prefix = "rewrite", output_dir = transcript_output_dir, output_ext = "txt")['output_file_path']
						rewrite_file_paths.append(rewrite_file_path)
						st.write(f"Rewriting {i}-th transcript out of total {len(transcript_file_paths)} transcripts.")
						for paragraph in paragraphs: 
							try:
								res = rewrite_chain({"functions": rewrite_instruction, "paragraph": paragraph}, return_only_outputs = True)
							except Exception as e:
								print(e)
								continue
							with open( rewrite_file_path, "a") as f:
								f.write(res["text"])
							st.write(res["text"][:50])
							st.divider()

				## --- summarize --- ##
				with st.spinner(f"summarize { ', '.join(rewrite_file_paths) }"):
					summary_text = ""
					for rewrite, rewrite_file_path in zip( rewrites, rewrite_file_paths):
						start_time = time.time()
						summary_file_path = compose_file_path( input_file_path = rewrite_file_path, output_prefix = "summary", output_dir = summary_output_dir, output_ext = "txt")['output_file_path']
						summary = text2summary( rewrite, summary_file_path, instruction = "give helpful and concise summary with action items in traditional chinese.")
						st.write(summary)
						st.divider()
						summary_text += f"{summary}\n"
						print( f"[summary] Elapsed time: {time.time() - start_time} seconds")
	with tab7:
		today = date.today()
		topn_results = 3
		topn_articles = 1
		trend_results = get_trending_searches(today)
		highlights = []
		for query in trend_results[0].values[:topn_results]:
			search_results = get_search_results( secret_key, query, str(today) )
			highlights += search_results['organic_results'][:topn_articles]
		highlights = pd.DataFrame(highlights)
		st.dataframe(highlights[['title', 'link', 'snippet', 'source']])


elif authentication_status == False:
	st.error('Username/password is incorrect')
	
elif authentication_status == None:
	st.warning('Please enter your username and password')

