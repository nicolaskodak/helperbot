"""Main entrypoint for the app."""
import os
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks
import urllib
import pprint
from fastapi import FastAPI, Request, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse,HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from fastapi_login import LoginManager #Loginmanager Class
from fastapi_login.exceptions import InvalidCredentialsException #Exception class
from langchain.vectorstores import VectorStore
from langchain.docstore.document import Document

from dotenv import load_dotenv
load_dotenv()

from schema import Question
# from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
# from query_data import get_chain, get_faq_chain, parse_answer
# from schemas import ChatResponse

from model import get_chain, get_vectorstore, get_response, get_answer
from postprocess import parse_url, parse_element

app = FastAPI()
# templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
chain = None


# SECRET = os.environ.get("SECRET") # "secret-key"
# manager = LoginManager(SECRET, token_url="/auth/login",use_cookie=True)
# manager.cookie_name = "cookiemonster"
# DB = { os.environ.get("USERNAME"):{"password": os.environ.get("PASSWORD")}} # unhashed


## ---- data ---- ##
CLIENT = 'data/pain'


@app.on_event("startup")
async def startup_event():
    # vector_store_path = f"{CLIENT}/vectorstore/vectorstore.pkl"
    # logging.info("loading vectorstore")
    # if not Path(vector_store_path).exists():
    #     raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    # with open( vector_store_path, "rb") as f:
    #     global vectorstore
    #     vectorstore = pickle.load(f)

    # print("Loading chain...")
    # chain = get_chain()
    pass


@app.get("/validate")
def validate( client_id: str, redirect_uri: str, state: str = "f094a459-1d16-42d6-a709-c2b61ec53d60"):
    """
    Argument
        - client_id: str
            e.g. 'KW52IGf7UP0vyrTVc8xINk'
        - redirect_uri: str
            e.g. 'https://d890-122-116-55-147.ngrok.io'
        - state: str (csrf token)
            e.g. "f094a459-1d16-42d6-a709-c2b61ec53d60"
    """
    # return templates.TemplateResponse("login.html", {"request": request})
    response_type = 'code'
    scope = 'notify'
    response_mode = 'form_post'
    validation_url = f"https://notify-bot.line.me/oauth/authorize?response_type={response_type}&scope={scope}&response_mode={response_mode}&client_id={client_id}&redirect_uri={redirect_uri}&state={state}"
    # return RedirectResponse(validation_url, status_code=status.HTTP_302_FOUND)
    return validation_url


# @manager.user_loader
# def load_user(username:str):
#     user = DB.get(username)
#     return user


# @app.post("/auth/login")
# def login(data: OAuth2PasswordRequestForm = Depends()):
#     username = data.username
#     password = data.password
#     user = load_user(username)
#     if not user:
#         raise InvalidCredentialsException
#     elif password != user['password']:
#         raise InvalidCredentialsException
#     access_token = manager.create_access_token(
#         data={"sub":username}
#     )
#     # resp = RedirectResponse(url="/private",status_code=status.HTTP_302_FOUND)
#     resp = RedirectResponse(url="/knowbot",status_code=status.HTTP_302_FOUND)
#     manager.set_cookie(resp,access_token)
#     return resp


# @app.get("/private")
# def getPrivateendpoint(_=Depends(manager)):
#     return "You are an authentciated user"


@app.post("/webqa")
def webqa(request: Question, background_tasks: BackgroundTasks):
    """
    """
    print(f"received { request.question }" )
    res = get_answer( request.question )
    print(f"webqa res -> {res}")
    return res



# @app.websocket("/chat")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     question_handler = QuestionGenCallbackHandler(websocket)
#     stream_handler = StreamingLLMCallbackHandler(websocket)
#     chat_history = []
#     qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)
#     faq_chain = get_faq_chain(stream_handler, tracing=True)
#     # Use the below line instead of the above line to enable tracing
#     # Ensure `langchain-server` is running
#     # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

#     result = None

#     while True:
#         try:
#             # -- Receive and send back the client message -- #
#             question = await websocket.receive_text()
#             resp = ChatResponse(sender="you", message=question, type="stream")
#             await websocket.send_json(resp.dict())

#             # -- Construct a response -- #
#             print(f"--- Start ---")
#             start_resp = ChatResponse(sender="bot", message="", type="start")
#             await websocket.send_json(start_resp.dict())

#             # -- LLM chat -- #
#             print(f"--- Async LLM Chat ---")
#             # print(f"\n##### chat_history = {chat_history}")
#             result = await qa_chain.acall( {"question": question, "chat_history": chat_history} ) # chatvectordb worked, but not worked after langchain is updated
#             # result = qa_chain._call(
#             #     {"question": question, "chat_history": chat_history}
#             # ) # error: aysnc call manager never awaited

#             # -- History (context) -- #
#             print(f"--- Add to history ---")
#             print(f"\n##### result = {result}")
#             chat_history.append((question, result["answer"]))

#             # -- show reference text -- #
#             # if "沒有找到最直接相關的答案" in result["answer"]:
#                 # print(f"--- Async LLM chat (in Exception) ---")
#                 # result = await qa_chain.acall( {"question": "繼續", "chat_history": chat_history} ) # chatvectordb worked, but not worked after langchain is updated

#                 # # -- History (context) -- #
#                 # print(f"--- Add to history (in Exception) ---")
#                 # print(f"\n##### result = {result}")
#                 # chat_history.append((question, result["answer"]))

#             page_contents = '\n-----\n'.join([  x.page_content for x in result['source_documents']])
#             print( f"\n##### source documents page content -> {page_contents }")
            
#             # -- return reference source -- #
#             # source = "<br>【參考資料】<br>" + "<br>".join( 
#             #     sorted(list(set([ "- " + x.metadata['source'].rsplit("/",1)[-1] for x in result['source_documents']])))
#             # ) # f"\n\n參考資料: {source}"
#             # ref_resp = ChatResponse(sender="bot", message=source, type="stream")
#             # await websocket.send_json(ref_resp.dict())

#             print( f"\n## --- parse answer --- ##")
#             parsed_answer = parse_answer( result['answer'])
#             print( f"\n##### parsed_answer -> {parsed_answer}")
#             # url = urllib.parse.quote(parsed_answer['url'],safe='://')
#             url = parsed_answer['url']
#             source = parsed_answer['source']
#             parsed_resp = ChatResponse(sender="bot", message=f"<br>【深入了解】<a href=\"{url}\">{source}</a>", type="stream")
#             await websocket.send_json(parsed_resp.dict())

#         except WebSocketDisconnect:
#             print(f"--- Disconnected ---")
#             print("\n@@@@@ websocket disconnect")
#             break
#         except Exception as e:
#             print(e)
#             # resp = ChatResponse(
#             #     sender="bot",
#             #     message="我想想喔～",
#             #     type="error",
#             # )
#             # await websocket.send_json(resp.dict())

#             # -- LLM chat -- #
#             print(f"\n@@@@@ ERROR# -> {e} -> will continue chat with previous question")
#             # logging.info(f"chat_history = {chat_history}")
#             print(f"\n##### chat_history (in exception)= {chat_history}")
#             print(f"\n##### result so far (in exception)= {result}")

#             print(f"--- Async LLM chat (in Exception) ---")
#             result = await qa_chain.acall( {"question": "請繼續。", "chat_history": chat_history} ) # chatvectordb worked, but not worked after langchain is updated

#             # # -- History (context) -- #
#             print(f"--- Add to history (in Exception) ---")
#             chat_history.append((question, result["answer"]))
#             # print(f"\n##### result = {result}")

#             # -- show reference text -- #
#             # page_contents = '\n'.join([  x.page_content for x in result['source_documents']])
#             # print( f"source documents page content -> {page_contents }")

#             # # -- end -- #
#             # print(f"--- End (in Exception) ---")
#             # end_resp = ChatResponse(sender="bot", message="", type="end")
#             # await websocket.send_json(end_resp.dict())
#         finally:
#             # -- return FAQ -- #
#             faq_head_resp = ChatResponse(sender="bot", message="<br>【你也許也想知道的問題】：", type="stream")
#             await websocket.send_json(faq_head_resp.dict())
#             if result and "沒有找到最直接相關的答案" in result["answer"]:
#                 faq_material = result['source_documents'][0].page_content
#                 # rele_content = vectorstore.similarity_search_with_score(query = question, k=1)
#                 # faq_material = rele_content[0][0].page_content
#             elif result:
#                 rele_content = vectorstore.similarity_search_with_score(query = result["answer"], k=1)
#                 faq_material = rele_content[0][0].page_content
#             else:
#                 rele_content = vectorstore.similarity_search_with_score(query = question, k=1)
#                 faq_material = rele_content[0][0].page_content

#             faq_result = await faq_chain.acall({
#                 "input_documents": [ Document(page_content = faq_material)] 
#             }) 
#             print(f"\n##### faq_result = {faq_result}")

#             # -- end -- #
#             print(f"--- End ---")
#             end_resp = ChatResponse(sender="bot", message="", type="end")
#             await websocket.send_json(end_resp.dict())

#     print(f"@@@@@ loop is broken")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001)
