"""
PaperBanana local HTTP server.

Launched as a subprocess by the plugin. Wraps PaperBanana's
process_parallel_candidates() behind a minimal FastAPI endpoint.

Usage:
    <paperbanana_venv>/bin/python paperbanana_server.py \
        --paperbanana-path /path/to/PaperBanana \
        --port 8765
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="PaperBanana Local Service")

_processor = None
_exp_config = None


class GenerateRequest(BaseModel):
    text: str
    caption: str = ""


class GenerateResponse(BaseModel):
    success: bool
    image_base64: str = ""
    error: str = ""


def _init_processor(paperbanana_path: str):
    """Import PaperBanana modules and create the processor."""
    global _processor, _exp_config

    pb_path = Path(paperbanana_path).resolve()
    if str(pb_path) not in sys.path:
        sys.path.insert(0, str(pb_path))

    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    from utils import config
    from utils.paperviz_processor import PaperVizProcessor

    _exp_config = config.ExpConfig(
        dataset_name="PluginDemo",
        split_name="demo",
        exp_mode="dev_planner_critic",
        retrieval_setting="auto",
        main_model_name="",
        image_gen_model_name="",
        work_dir=pb_path,
    )

    _processor = PaperVizProcessor(
        exp_config=_exp_config,
        vanilla_agent=VanillaAgent(exp_config=_exp_config),
        planner_agent=PlannerAgent(exp_config=_exp_config),
        visualizer_agent=VisualizerAgent(exp_config=_exp_config),
        stylist_agent=StylistAgent(exp_config=_exp_config),
        critic_agent=CriticAgent(exp_config=_exp_config),
        retriever_agent=RetrieverAgent(exp_config=_exp_config),
        polish_agent=PolishAgent(exp_config=_exp_config),
    )


async def _generate_image(text: str, caption: str) -> str:
    """Run a single PaperBanana generation and return base64 PNG."""
    input_data = {
        "filename": f"plugin_{uuid.uuid4().hex[:8]}",
        "caption": caption or "Methodology Diagram",
        "content": text,
        "visual_intent": caption or text[:200],
        "additional_info": {"rounded_ratio": "16:9"},
        "max_critic_rounds": 1,
    }

    results = []
    async for result_data in _processor.process_queries_batch(
        [input_data], max_concurrent=1, do_eval=False
    ):
        results.append(result_data)

    if not results:
        raise ValueError("PaperBanana returned no results")

    result = results[0]
    task_name = "diagram"

    # Try to find the best image: critic rounds → stylist → planner
    image_b64 = None
    for r in range(3, 0, -1):
        key = f"target_{task_name}_critic_desc{r}_base64_jpg"
        if result.get(key):
            image_b64 = result[key]
            break
    if not image_b64:
        key = f"target_{task_name}_stylist_desc0_base64_jpg"
        image_b64 = result.get(key)
    if not image_b64:
        key = f"target_{task_name}_desc0_base64_jpg"
        image_b64 = result.get(key)

    if not image_b64:
        raise ValueError("No image found in PaperBanana result")

    # Decode JPG base64 → re-encode as PNG for consistency
    jpg_bytes = base64.b64decode(image_b64)
    from PIL import Image
    from io import BytesIO
    img = Image.open(BytesIO(jpg_bytes))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        print(f"[paperbanana_server] Generating image (text_len={len(req.text)}, caption={req.caption[:60]})...", flush=True)
        img_b64 = await _generate_image(req.text, req.caption)
        print(f"[paperbanana_server] Image generated successfully ({len(img_b64)} chars base64)", flush=True)
        return GenerateResponse(success=True, image_base64=img_b64)
    except Exception as e:
        print(f"[paperbanana_server] Generation failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paperbanana-path", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"[paperbanana_server] Initializing from: {args.paperbanana_path}", flush=True)
    _init_processor(args.paperbanana_path)
    print(f"[paperbanana_server] Ready on port {args.port}", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
