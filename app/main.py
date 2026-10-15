import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .graph import graph
from .schemas import CheckRequest, FactCheckReport
from .tools.fetch import FetchError

app = FastAPI(title="Fact-check agent", version="0.1.0")


def _initial(req: CheckRequest) -> dict:
    return {
        "url": str(req.url) if req.url else None,
        "raw_text": req.text,
        "model_provider": req.model_provider,
        "model": req.model,
        "limitations": [],
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/check", response_model=FactCheckReport)
async def check(req: CheckRequest):
    try:
        final = await graph.ainvoke(_initial(req))
    except FetchError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return final["report"]


@app.post("/check/stream")
async def check_stream(req: CheckRequest):
    """Server-sent events: one 'progress' event per finished node, then 'report'."""

    async def gen():
        try:
            async for chunk in graph.astream(_initial(req), stream_mode="updates"):
                for node, update in chunk.items():
                    yield f"event: progress\ndata: {json.dumps({'node': node})}\n\n"
                    if node == "synthesize":
                        yield f"event: report\ndata: {update['report'].model_dump_json()}\n\n"
        except FetchError as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
