# main.py
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.query import Query
from models.coursesupdated import CoursesUpdated
from chat_moodle import ChatMoodle
from moodle_store import MoodleStore
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
@app.post("/api/v1/query")
async def query(query: Query):
    if query.token != os.getenv("API_KEY"):
        raise HTTPException(status_code=404, detail="Permission denied. Invalid token.")

    # Initialize chat engine.
    chat_moodle = ChatMoodle(token=os.getenv("WS_TOKEN"), llm_provider=query.llm_provider, model_name=query.model_name,
    model_cache=str(os.getenv("MODEL_CACHE")), max_tokens=query.max_call_per_day, doc_language=query.doc_language,
    courseid=query.courseid, instruction=query.instruction, history=query.history)

    # Call chat engine.
    answer = chat_moodle.call_chat(query.message)

    # Answer
    return {"answer": answer}

@app.post("/api/v1/coursesupdated")
async def coursesupdated(coursesupdated: CoursesUpdated):
    if coursesupdated.token != os.getenv("API_KEY"):
        raise HTTPException(status_code=404, detail="Permission denied. Invalid token.")
    # Launch moodle-lmsassistant.py
    moodle_store = MoodleStore(os.getenv("WS_TOKEN"), os.getenv("WS_ENDPOINT"), os.getenv("WS_STORAGE"))
    moodle_store.store()

    return {"coursesupdated":"coursesupdated"}