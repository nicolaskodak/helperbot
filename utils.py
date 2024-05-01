import os
import argparse
import time
import math
import toml
import logging

from pydub import AudioSegment
import openai
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
# from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
import streamlit as st

# from dotenv import load_dotenv
# load_dotenv()
# os.environ.get("OPENAI_API_KEY")
# openai.api_key = os.environ.get("OPENAI_API_KEY")

########### ===== config ===== #############
# config_path = '.streamlit/secrets.toml'
# if os.path.exists(config_path):
# 	print(f"[utils] {config_path} exists")
# 	config = toml.load(open( config_path, 'r'))
# else:
# 	print( f"[utils] secrets -> {st.secrets}" )
# 	config = dict(st.secrets.items())
# print( f"[utils] config -> {config}")
# for k in ['name', 'authentication_status', 'username' ]:
# 	st.session_state[k] = None

# openai.api_key = os.environ.get("OPENAI_API_KEY") or config["settings"]["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai.api_key
########### ==================== #############

from openai import OpenAI as OpenAIClient
client = OpenAIClient()

def split( in_file_path:str, length:int=10) -> list:
	"""
	Argument
		in_file_path: str
		length: int (in minutes)
	"""
	dirname = os.path.dirname( in_file_path)
	basename = os.path.basename( in_file_path)
	filename, ext = os.path.splitext( basename)
	ext = ext[1:]
	sound = AudioSegment.from_file(in_file_path)
	ext_format = 'ipod' if ext == 'm4a' else ext
	# PyDub handles time in milliseconds
	duration = length * 60 * 1000
	n_segments = math.ceil( len(sound) / duration)
	output_file_paths = []
	for i in range(n_segments):
		print(f"i -> {i}")
		segment = sound[  i*duration: (i+1)*duration]
		out_file_path = f"{dirname}/{basename}_{i}.{ext}"
		# if os.path.exists(out_file_path):
		# 	print(f"Output file path {out_file_path} already exists. Will skip this file.")
		# 	continue
		try:
			segment.export( out_file_path, format=ext_format)
			print(f"output file path -> {out_file_path}")
			output_file_paths.append( out_file_path)
		except Exception as e:
			print(f"Error: {e} -> fails to export {out_file_path}")
	return output_file_paths


def compose_file_path( input_file_path: str, output_dir: str, output_prefix: str = None, output_ext: str = None):
	"""
	Argument
		input_file_path: str
		output_dir: str
		output_prefix: str
		output_ext: str

	"""
	filename = os.path.basename(input_file_path)
	fn, ext = os.path.splitext(filename)
	if output_prefix:
		fn = f"{fn}_{output_prefix}"
	if output_ext:
		output_file_path = f"{output_dir}/{fn}.{output_ext}"
	else:
		output_file_path = f"{output_dir}/{fn}.{ext[1:]}"
	return { 'output_file_path': output_file_path, 'input_filename': filename, 'input_fn': fn, 'input_ext': ext[1:]}


def get_summarize_chain():
	"""
	"""
	sum_template = '''Given a piece of text delimited by triple quotes. Your taks is to {functions}:

	"""{text}"""

	SUMMARY: '''
	sum_prompt = PromptTemplate(template = sum_template, input_variables=["text", "functions"])
	sum_llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct", max_tokens = 1024)
	# chain = load_qa_with_sources_chain( qa_llm, prompt = qa_prompt, document_variable_name = "context", chain_type="stuff")
	# chain = load_summarize_chain( sum_llm, prompt = sum_prompt, chain_type="map_reduce", return_intermediate_steps=True)
	chain = load_summarize_chain( sum_llm, prompt = sum_prompt, chain_type="stuff")
	return chain


def get_rewrite_chain():
	rewrite_template = '''A paragraph delimited by triple quotes will be provided.
	Your task is to rewrite this paragraph in order to {functions}. 
	Please keep its original meaning, style and details. Do not make up any information. The rewrite is provided in traditional Chinese.


	Paragraph: """{paragraph}"""


	Rewrite:'''

	rewrite_prompt = PromptTemplate(template = rewrite_template, input_variables=["paragraph", "functions"])
	rewrite_llm = OpenAI( temperature=0, model="gpt-3.5-turbo-instruct", max_tokens = 1024)

	# chain = load_qa_with_sources_chain( qa_llm, prompt = qa_prompt, document_variable_name = "context", chain_type="stuff")
	rewrite_chain = LLMChain( llm=rewrite_llm, prompt = rewrite_prompt)
	return rewrite_chain


# def get_response( chain, docs: list, question: str):
#     """
#     Argument
#     --------
#     chain: langchain.chains.qa_with_sources.QAWithSourcesChain
#     docs: list
#     question: str
#         e.g. "肩膀不太舒服，前幾天有去打球，會不會是韌帶拉傷？"
#     """
#     # res = chain({"input_documents": rele_docs, "question": question}, return_only_outputs=False)
#     res = chain({"input_documents": docs}, return_only_outputs=False)
#     print( res )
#     return res

def sound2txt( audio_file_path, transcript_file_path: str):
	"""
	Arguments
		audio_file_path: str
		transcript_file_path: str

	Response
		transcript: dict (see https://beta.openai.com/docs/api-reference/transcriptions/create)
	"""
	with open( audio_file_path, "rb") as audio_file:
		try:
			# transcript = openai.Audio.transcribe("whisper-1", audio_file)
			transcript = client.audio.transcriptions.create( model="whisper-1", file=audio_file)
		except openai.error.InvalidRequestError:
			print( f"InvalidRequestError: {audio_file_path}")
		with open( transcript_file_path, "w") as f:
			f.write(transcript.text)
		return transcript
	

def text2summary( article: str, summary_file_path: str, instruction: str, chunk_size: int = 1024, chunk_overlap: int = 128):
	"""
	Arguments
		article: str
		summary_file_path: str
		instruction: str
		chunk_size: int
		chunk_overlap: int
	"""
	summarize_chain = get_summarize_chain()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size = chunk_size,
		chunk_overlap = chunk_overlap,
	)
	paragraphs = text_splitter.split_text(article)
	summary_text = ""
	with open( summary_file_path, "w") as f:
		for paragraph in paragraphs: 
			try:
				res = summarize_chain( {"input_documents": [ Document( page_content=paragraph) ], "functions": instruction} )
				summary = res['output_text']
			except Exception as e:
				print(e)
				continue
			f.write(summary)
			summary_text += summary + "\n"
	return summary_text