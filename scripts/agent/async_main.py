import asyncio
import time
import json
import sys
import random
import os
from concurrent.futures import ThreadPoolExecutor
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.columns import Columns
import questionary
from typing import List

from ingest import IngestManager
from schemas import POIRecord, AgentPrediction
from config import config
from llm.interface import GroqLLM
from agent_tools import execute_tool
from pydantic import BaseModel, Field

console = Console()

class DashboardState:
    def __init__(self):
        self.logs = [] 
        self.thoughts = "" 
        self.target_thought = "Initializing pipeline..." 
        self.last_results = [] 
        self.error_logs = []   
        self.research_count = 0
        self.inference_count = 0
        self.total = 0

state = DashboardState()

async def ui_tick_task():
    while True:
        if len(state.thoughts) < len(state.target_thought):
            state.thoughts += state.target_thought[len(state.thoughts)]
        elif len(state.thoughts) > len(state.target_thought):
            state.thoughts = state.target_thought[:len(state.thoughts)-1]
        await asyncio.sleep(0.01)

def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=5)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1)
    )
    layout["left"].split_column(
        Layout(name="thought_box", ratio=2),
        Layout(name="log_box", ratio=2),
        Layout(name="error_box", ratio=1)
    )
    return layout

def update_layout(layout, progress_renderable):
    layout["header"].update(Panel(Text.from_markup("[bold cyan]StatusNow V6 Async Control Center[/bold cyan]", justify="center"), border_style="blue"))
    layout["thought_box"].update(Panel(Text(state.thoughts, style="italic green"), title="[bold magenta]AI Reasoning Stream[/bold magenta]", border_style="magenta"))
    layout["log_box"].update(Panel("\n".join(state.logs[-6:]), title="[bold yellow]Research Activity Log[/bold yellow]", border_style="yellow"))
    layout["error_box"].update(Panel("\n".join(state.error_logs[-3:]), title="[bold red]Inference Errors[/bold red]", border_style="red"))
    
    res_table = Table(expand=True, box=None)
    res_table.add_column("POI", style="cyan")
    res_table.add_column("Status", style="bold")
    for res in state.last_results[-12:]:
        st = res['status']
        style = "green" if st == "OPEN" else ("yellow" if st == "TIMEOUT" else "red")
        res_table.add_row(res['name'][:30], f"[{style}]{st}[/{style}]")
    layout["right"].update(Panel(res_table, title="[bold cyan]Inference Stream[/bold cyan]", border_style="cyan"))
    layout["footer"].update(Panel(progress_renderable, border_style="dim white"))

class FastPrediction(BaseModel):
    predicted_label: int = Field(description="1 if the POI is Open, 0 if it is Closed.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Brief explanation of your reasoning.")

class FastPredictionWithId(FastPrediction):
    poi_id: str

class BatchFastPrediction(BaseModel):
    predictions: List[FastPredictionWithId] = Field(description="List of predictions for each POI in the batch.")

async def research_worker(poi_queue, inference_queue, progress, task_research, tavily_client):
    executor = ThreadPoolExecutor(max_workers=5)
    loop = asyncio.get_running_loop()
    while True:
        try:
            poi = poi_queue.get_nowait()
        except asyncio.QueueEmpty: break
        t_query = f"{poi.name} {poi.address or ''} permanently closed"
        state.logs.append(f"🌐 [dim]Tavily[/dim]: Researching {poi.name}...")
        try:
            t_resp = await loop.run_in_executor(executor, lambda: tavily_client.search(query=t_query, max_results=3))
            state.logs.append(f"✅ [dim]Tavily[/dim]: Found data for {poi.name}")
        except Exception: t_resp = {"error": "Research failed"}
        state.logs.append(f"🔍 [dim]Yelp[/dim]: Enriching {poi.name}...")
        try:
            y_resp = await loop.run_in_executor(executor, lambda: execute_tool(tavily_client, "yelp_search", {"term": poi.name, "location": poi.address or ""}))
            state.logs.append(f"✨ [dim]Yelp[/dim]: Bundled {poi.name}")
        except Exception: y_resp = {"error": "Yelp failed"}
        await inference_queue.put({"poi": poi, "tavily_data": t_resp, "yelp_data": y_resp})
        poi_queue.task_done()
        progress.update(task_research, advance=1)
    executor.shutdown(wait=False)

async def inference_worker(inference_queue, results_list, progress, task_inference, llm):
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    MAX_BATCH = 5
    current_batch = []

    async def flush_batch(batch):
        if not batch: return
        names = ", ".join([p["poi"].name for p in batch])
        sys_p = "You are an expert status predictor. Analyze the following batch and return a JSON list of predictions (1=OPEN, 0=CLOSED)."
        batch_data = []
        for p in batch:
            batch_data.append({
                "poi_id": p["poi"].poi_id, "name": p["poi"].name, "addr": p["poi"].address,
                "web": json.dumps(p['tavily_data'])[:1200], "yelp": json.dumps(p['yelp_data'])[:1200]
            })
        u_p = f"Batch Analysis ({len(batch)} POIs):\n{json.dumps(batch_data, indent=2)}"
        
        for attempt in range(4):
            state.target_thought = f"🚀 Batch Inference (Attempt {attempt+1}): {len(batch)} items..."
            try:
                out = await asyncio.wait_for(loop.run_in_executor(executor, lambda: llm.generate_structured_output(sys_p + "\n\n" + u_p, BatchFastPrediction)), timeout=60)
                p_map = {p.poi_id: p for p in out.predictions}
                for item in batch:
                    poi = item["poi"]
                    pred = p_map.get(poi.poi_id)
                    st = "OPEN" if (pred and pred.predicted_label == 1) else ("CLOSED" if pred else "ERROR")
                    results_list.append({"poi_id": poi.poi_id, "name": poi.name, "status": st, "prediction": pred.model_dump() if pred else {}})
                    state.last_results.append({"name": poi.name, "status": st})
                    progress.update(task_inference, advance=1, description=f"[cyan]Infer[/cyan]: {poi.name} -> {st}")
                state.target_thought = f"✅ Success: Processed batch of {len(batch)}."
                return
            except Exception as e:
                err = str(e)
                if any(x in err.lower() for x in ["429", "rate_limit"]):
                    wait = 30 * (attempt + 1)
                    state.target_thought = f"⚠️ Rate limited. Cooling down for {wait}s..."
                    state.error_logs.append(f"⏳ Cooldown ({wait}s) due to Groq limits.")
                    await asyncio.sleep(wait)
                    continue
                state.error_logs.append(f"❌ Batch Error: {err[:80]}")
                for item in batch:
                    poi = item["poi"]
                    results_list.append({"poi_id": poi.poi_id, "name": poi.name, "status": "ERROR"})
                    state.last_results.append({"name": poi.name, "status": "ERROR"})
                    progress.update(task_inference, advance=1)
                break

    while True:
        try:
            payload = await asyncio.wait_for(inference_queue.get(), timeout=5.0 if current_batch else None)
            if payload is None:
                await flush_batch(current_batch)
                inference_queue.task_done()
                break
            current_batch.append(payload)
            state.target_thought = f"📥 Buffering: {len(current_batch)}/{MAX_BATCH} POIs ready..."
            if len(current_batch) >= MAX_BATCH:
                await flush_batch(current_batch)
                current_batch = []
            inference_queue.task_done()
        except asyncio.TimeoutError:
            if current_batch:
                await flush_batch(current_batch)
                current_batch = []
    executor.shutdown(wait=False)

async def async_pipeline():
    config.validate_keys()
    console.print(Panel("[bold cyan]StatusNow V6 Async Pipeline[/bold cyan]", border_style="cyan"))
    with console.status("[dim]Loading data...[/dim]"):
        man = IngestManager("data/v5_predictions_export.parquet", 0.65)
        all_pois = man.load_and_filter()
    
    sel = await questionary.select("Count:", choices=["10","20","50"]).ask_async()
    num = int(sel or 10)
    pois = random.sample(all_pois, min(num, len(all_pois)))
    
    mod = await questionary.select("Model:", choices=["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"]).ask_async()
    llm = GroqLLM(mod or "llama-3.3-70b-versatile")
    
    p_q, i_q = asyncio.Queue(), asyncio.Queue()
    results_list = []
    for p in pois: p_q.put_nowait(p)
    
    prog = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console, expand=True)
    t_r = prog.add_task("[yellow]Researching...[/yellow]", total=len(pois))
    t_i = prog.add_task(f"[cyan]Inference...[/cyan]", total=len(pois))
    
    layout = make_layout()
    anim = asyncio.create_task(ui_tick_task())
    with Live(layout, refresh_per_second=10, screen=True):
        workers = [asyncio.create_task(research_worker(p_q, i_q, prog, t_r, config.tavily_api_key)) for _ in range(3)]
        inf = asyncio.create_task(inference_worker(i_q, results_list, prog, t_i, llm))
        
        from tavily import TavilyClient
        tc = TavilyClient(api_key=config.tavily_api_key)
        # Update research workers with real client
        for w in workers: w.cancel()
        workers = [asyncio.create_task(research_worker(p_q, i_q, prog, t_r, tc)) for _ in range(3)]
        
        while not p_q.empty() or not i_q.empty() or any(not w.done() for w in workers) or not inf.done():
            update_layout(layout, prog)
            await asyncio.sleep(0.2)
            if p_q.empty() and all(w.done() for w in workers) and i_q.empty(): break
        
        await i_q.put(None)
        await asyncio.gather(*workers, inf, return_exceptions=True)
    
    anim.cancel()
    console.print("\n", Panel("[bold green]🏁 Pipeline Execution Complete[/bold green]", border_style="green", expand=False))
    
    # Render final results as a clean list instead of a table
    results_display = []
    for r in results_list:
        status = r["status"]
        style = "green" if status == "OPEN" else "red"
        icon = "●" if status == "OPEN" else "○"
        results_display.append(f"[{style}]{icon} {r['name'].ljust(45)}[/][bold {style}]{status}[/]")
    
    console.print(Panel("\n".join(results_display), title="[bold white]Final Batch Results[/bold white]", border_style="cyan", padding=(1, 2)))

if __name__ == "__main__":
    try: asyncio.run(async_pipeline())
    except KeyboardInterrupt: sys.exit(0)
