import asyncio
import time
import json
import sys
import random
from concurrent.futures import ThreadPoolExecutor
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.layout import Layout
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import questionary

from ingest import IngestManager
from schemas import POIRecord, AgentPrediction
from config import config
from llm.interface import GroqLLM
from agent_tools import execute_tool
from pydantic import BaseModel, Field

console = Console()

class FastPrediction(BaseModel):
    predicted_label: int = Field(description="1 if the POI is Open, 0 if it is Closed.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Brief explanation of your reasoning.")

def print_banner():
    panel = Panel(
        "[bold cyan]StatusNow V6 Async Agent[/bold cyan]\n[italic blue]Decoupled Research & Inference Pipeline[/italic blue]",
        border_style="cyan",
        expand=False,
        padding=(1, 4)
    )
    console.print(panel, justify="center")
    console.print()

async def research_worker(poi_queue, inference_queue, progress, task_research, tavily_client):
    executor = ThreadPoolExecutor(max_workers=5)
    loop = asyncio.get_running_loop()
    
    while True:
        try:
            poi = poi_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
            
        # Execute research asynchronously using threadpool
        tavily_query = f"{poi.name} {poi.address or ''} {poi.category or ''} permanently closed"
        
        # 1. Web Search
        try:
            tavily_resp = await loop.run_in_executor(
                executor, 
                lambda: tavily_client.search(query=tavily_query, max_results=3)
            )
        except Exception as e:
            tavily_resp = {"error": str(e)}
            
        # 2. Yelp Search & Enrich
        try:
            yelp_resp = await loop.run_in_executor(
                executor,
                lambda: execute_tool(tavily_client, "yelp_search", {"term": poi.name, "location": poi.address or ""})
            )
        except Exception as e:
            yelp_resp = {"error": str(e)}
            
        research_payload = {
            "poi": poi,
            "tavily_data": tavily_resp,
            "yelp_data": yelp_resp
        }
        
        await inference_queue.put(research_payload)
        poi_queue.task_done()
        progress.update(task_research, advance=1)
        
    executor.shutdown(wait=False)

async def inference_worker(inference_queue, results_list, progress, task_inference, llm):
    executor = ThreadPoolExecutor(max_workers=5)
    loop = asyncio.get_running_loop()
    
    while True:
        payload = await inference_queue.get()
        if payload is None: # Sentinel value
            inference_queue.task_done()
            break
            
        poi = payload["poi"]
        
        system_prompt = "You are an AI tasked with predicting if a Point of Interest (POI) is open or closed."
        user_prompt = f"""
POI Name: {poi.name}
Address: {poi.address}
Category: {poi.category}

Web Search Results:
{json.dumps(payload['tavily_data'])[:2000]}

Yelp Directory Data:
{json.dumps(payload['yelp_data'])[:2000]}

Given this context, output a prediction for the status of the business. You MUST guess 1 or 0.
"""
        
        prompt = system_prompt + "\n\n" + user_prompt
        
        try:
            prediction = await loop.run_in_executor(
                executor,
                lambda: llm.generate_structured_output(prompt, FastPrediction)
            )
            status_str = "OPEN" if prediction.predicted_label == 1 else "CLOSED"
        except Exception as e:
            prediction = None
            status_str = "ERROR"
            
        results_list.append({
            "poi_id": poi.poi_id,
            "status": status_str,
            "prediction": prediction.model_dump() if prediction else {"error": str(e)}
        })
        
        inference_queue.task_done()
        progress.update(task_inference, advance=1, description=f"[cyan]Inference[/cyan]: {poi.name} -> {status_str}")
        
    executor.shutdown(wait=False)

def select_pois_phase(all_flagged_pois):
    if not all_flagged_pois:
        console.print("[red]No POIs available to select.[/red]")
        return []
        
    choice = questionary.select(
        "How many POIs would you like to process asynchronously?",
        choices=[
            "1. Process 10 random POIs",
            "2. Process 50 random POIs",
            "3. Process 100 random POIs"
        ]
    ).ask()
    
    if not choice:
        sys.exit(0)
        
    if choice.startswith("1"):
        return random.sample(all_flagged_pois, min(10, len(all_flagged_pois)))
    elif choice.startswith("2"):
        return random.sample(all_flagged_pois, min(50, len(all_flagged_pois)))
    elif choice.startswith("3"):
        return random.sample(all_flagged_pois, min(100, len(all_flagged_pois)))
    return []

async def async_pipeline():
    config.validate_keys()
    print_banner()
    
    # Ingestion
    with console.status("[bold cyan]Loading POIs from parquet...[/bold cyan]"):
        manager = IngestManager("data/v5_predictions_export.parquet", 0.65)
        all_flagged_pois = manager.load_and_filter()
        
    selected_pois = select_pois_phase(all_flagged_pois)
    if not selected_pois:
        return
        
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=config.tavily_api_key)
    llm = GroqLLM("llama-3.3-70b-versatile")
    # For cheaper fast inference, we could use llama-3-8b, but let's stick to 70b as the user has keys.
    
    poi_queue = asyncio.Queue()
    inference_queue = asyncio.Queue()
    results_list = []
    
    for poi in selected_pois:
        poi_queue.put_nowait(poi)
        
    num_pois = len(selected_pois)
    
    # Setup rich progress layout
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        expand=True
    )
    
    task_research = progress.add_task("[yellow]Researching POIs via APIs...[/yellow]", total=num_pois)
    task_inference = progress.add_task("[cyan]Pending Inference...[/cyan]", total=num_pois)
    
    console.print(f"\n[bold magenta]Launching Asynchronous Pipeline for {num_pois} POIs...[/bold magenta]")
    
    with Live(progress, refresh_per_second=10):
        # Spawn 3 Research workers (IO heavy)
        researchers = [
            asyncio.create_task(research_worker(poi_queue, inference_queue, progress, task_research, tavily_client))
            for _ in range(5)
        ]
        
        # Spawn 2 Inference workers (LLM intensive)
        inferencers = [
            asyncio.create_task(inference_worker(inference_queue, results_list, progress, task_inference, llm))
            for _ in range(3)
        ]
        
        # Wait for all research to complete pushing to inference queue
        await poi_queue.join()
        
        # Tell inferencers no more data is coming
        for _ in inferencers:
            await inference_queue.put(None)
            
        # Wait for inference queue to drain completely
        await inference_queue.join()
        
        # Await tasks to close cleanly
        await asyncio.gather(*researchers, *inferencers)
        
    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    
    # Print summary table
    table = Table(title="Inference Results", style="bold green")
    table.add_column("POI ID", style="cyan")
    table.add_column("Predicted Status", style="magenta")
    
    for r in results_list:
        table.add_row(r["poi_id"], r["status"])
        
    console.print(table)

def main():
    try:
        asyncio.run(async_pipeline())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
        sys.exit(0)

if __name__ == "__main__":
    main()
