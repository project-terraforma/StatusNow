import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich import print as rprint
import sys
import random
from ingest import IngestManager
from planner import AgentPlanner
from executor import AgentExecutor
from config import config
from google import genai
from tavily import TavilyClient
from llm.interface import GeminiLLM, GroqLLM
from schemas import AgentPlan
import questionary

console = Console()

def print_banner():
    """Prints a beautiful welcome banner for the StatusNow V6 Agent."""
    title = Text("StatusNow V6 Agent", style="bold cyan", justify="center")
    subtitle = Text("AI-Powered POI Resolution Pipeline", style="italic blue", justify="center")
    
    panel = Panel(
        Text.assemble(title, "\n", subtitle),
        border_style="cyan",
        expand=False,
        padding=(1, 4)
    )
    console.print(panel, justify="center")
    console.print()

def run_ingest_phase(data_path="data/v5_predictions_export.parquet", threshold=0.65):
    """Phase 1: Ingest and filtering."""
    console.print("[bold green]Phase 1:[/bold green] Ingestion & Filtering")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task1 = progress.add_task("[cyan]Loading V5 output parquet...", total=100)
        manager = IngestManager(data_path, threshold)
        flagged_pois = manager.load_and_filter()
        
        for i in range(100):
            time.sleep(0.01)
            progress.update(task1, advance=1)
            

    # Summary table for ingest
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("POIs Flagged (Confidence < 0.65)", str(len(flagged_pois)))
    
    console.print(Panel(table, title="Ingestion Summary", border_style="green", expand=False))
    console.print()
    return flagged_pois

def select_pois_phase(all_flagged_pois):
    if not all_flagged_pois:
        console.print("[red]No POIs available to select.[/red]")
        return []
        
    console.print("\n[bold green]Phase 1.5:[/bold green] POI Selection")
    
    choice = questionary.select(
        "How many POIs would you like to process?",
        choices=[
            "1. Pick 1 random POI for deep analysis",
            "2. Pick 10 random POIs for broad analysis",
            "3. Pick custom POIs (Select with Space/Arrow Keys)"
        ],
        style=questionary.Style([('pointer', 'cyan bold'), ('selected', 'green bold')])
    ).ask()
    
    if not choice:
        sys.exit(0)
        
    if choice.startswith("1"):
        selected = random.sample(all_flagged_pois, min(1, len(all_flagged_pois)))
        console.print(f"[green]Selected {len(selected)} random POI(s).[/green]")
        return selected
    elif choice.startswith("2"):
        selected = random.sample(all_flagged_pois, min(10, len(all_flagged_pois)))
        console.print(f"[green]Selected {len(selected)} random POIs.[/green]")
        return selected
    elif choice.startswith("3"):
        return custom_poi_selection(all_flagged_pois)

def custom_poi_selection(all_flagged_pois):
    selected_pois = []
    page = 0
    page_size = 10
    total_pages = (len(all_flagged_pois) + page_size - 1) // page_size
    if total_pages == 0:
        return []

    while True:
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(all_flagged_pois))
        current_page_pois = all_flagged_pois[start_idx:end_idx]
        
        choices = []
        
        # TOP BORDER
        choices.append(questionary.Choice(
            title=[("class:border", "┌" + "─"*72 + "┐")], 
            value={"type": "sep"}, 
            disabled=""
        ))
        
        # POI ITEMS
        for poi in current_page_pois:
            is_selected = any(p.poi_id == poi.poi_id for p in selected_pois)
            check_str = "[★]" if is_selected else "[ ]"
            
            # Format to exactly 70 columns inner
            name_str = f"{poi.name[:24]:<24}" 
            cat_str = f"| {poi.category[:19]:<19}"
            conf_str = f"| Conf: {poi.original_confidence:.2f}"
            
            title = [
                ("class:border", "│ "),
                ("class:check", f"{check_str} "),
                ("class:name", f"{name_str} "),
                ("class:cat", f"{cat_str} "),
                ("class:conf", f"{conf_str:<19}"),
                ("class:border", "│")
            ]
            choices.append(questionary.Choice(title=title, value={"type": "poi", "poi": poi}))
            
        # MIDDLE BORDER
        choices.append(questionary.Choice(
            title=[("class:border", "├" + "─"*72 + "┤")], 
            value={"type": "sep"}, 
            disabled=""
        ))
        
        # NAVIGATION BOTTOM
        if page < total_pages - 1:
            choices.append(questionary.Choice(
                title=[("class:border", "│ "), ("class:nav", f"{'▶ Next Page':<70}"), ("class:border", " │")], 
                value={"type": "next"}
            ))
        if page > 0:
            choices.append(questionary.Choice(
                title=[("class:border", "│ "), ("class:nav", f"{'◀ Previous Page':<70}"), ("class:border", " │")], 
                value={"type": "prev"}
            ))
            
        choices.append(questionary.Choice(
            title=[("class:border", "│ "), ("class:done", f"{'🚀 Finish & Generate Plan':<70}"), ("class:border", " │")], 
            value={"type": "done"}
        ))
        
        # BOTTOM BORDER
        choices.append(questionary.Choice(
            title=[("class:border", "└" + "─"*72 + "┘")], 
            value={"type": "sep"}, 
            disabled=""
        ))

        style = questionary.Style([
            ('check', 'bold green'),
            ('name', 'bold magenta'),
            ('cat', 'cyan'),
            ('conf', 'yellow'),
            ('nav', 'bold yellow'),
            ('done', 'bold green'),
            ('border', 'dim white'),
            ('pointer', 'bold cyan'),
            ('selected', 'reverse')
        ])
        
        console.clear()
        console.print(f"\n[bold cyan]Custom POI Selection - Page {page+1}/{total_pages}[/bold cyan] [dim](Selected globally: {len(selected_pois)})[/dim]")
        
        choice = questionary.select(
            "Press Enter to toggle POIs or navigate:",
            choices=choices,
            style=style,
            instruction="(Arrow keys to move, Enter to select)"
        ).ask()
        
        if choice is None:
            sys.exit(0)
            
        if choice["type"] == "poi":
            poi = choice["poi"]
            if any(p.poi_id == poi.poi_id for p in selected_pois):
                selected_pois = [p for p in selected_pois if p.poi_id != poi.poi_id]
            else:
                selected_pois.append(poi)
        elif choice["type"] == "next":
            page += 1
        elif choice["type"] == "prev":
            page -= 1
        elif choice["type"] == "done":
            if not selected_pois:
                console.print("\n[yellow]No POIs selected. Defaulting to 1 random.[/yellow]")
                return random.sample(all_flagged_pois, min(1, len(all_flagged_pois)))
            console.print(f"\n[green]Selected {len(selected_pois)} custom POI(s).[/green]")
            return selected_pois

def print_plan(agent_plan):
    table = Table(title="Generated Enrichment Plan", show_header=True, header_style="bold yellow")
    table.add_column("Group", style="cyan", justify="center")
    table.add_column("Strategy", style="magenta")
    table.add_column("POI Count", justify="right", style="green")
    table.add_column("Est. Credits", justify="right", style="red")

    for group in agent_plan.groups:
        table.add_row(
            group.group_id, 
            str(group.strategy), 
            str(len(group.poi_ids)), 
            str(len(group.poi_ids) * group.max_results)
        )
    
    table.add_section()
    table.add_row("Total", "", str(agent_plan.total_pois), str(agent_plan.total_estimated_credits), style="bold")

    console.print(table)
    
    warning_text = Text(f"\nEstimated credits ({agent_plan.total_estimated_credits}) are within the 4,000 monthly budget limit.", style="italic green")
    console.print(warning_text)
    console.print()


def run_plan_phase(flagged_pois: list):
    """
    Phase 2: Generate or regenerate the investigation plan via Gemini.
    """
    console.print("\n[bold green]Phase 2:[/bold green] Plan Generation (Groq/Llama-3 API)")
    
    with console.status("[bold yellow]LLM is analyzing POIs and generating research strategies...", spinner="dots"):
        llm = GroqLLM("llama-3.3-70b-versatile")
        planner = AgentPlanner(llm=llm)
        agent_plan = planner.generate_plan(flagged_pois)

    print_plan(agent_plan)
    return agent_plan

def user_approval_gate(agent_plan):
    """Phase 3: Human Approval Gate."""
    console.print("\n[bold red]Phase 3:[/bold red] Human Approval Gate")
    
    choice = questionary.select(
        "Please review the suggested plan. What would you like to do?",
        choices=[
            questionary.Choice("✅ Approve plan and begin execution", value="APPROVE"),
            questionary.Choice("🧠 View LLM reasoning and strategy logic", value="REASONING"),
            questionary.Choice("✏️  Edit strategies or drop POIs", value="EDIT"),
            questionary.Choice("❌ Reject plan and exit", value="REJECT")
        ],
        style=questionary.Style([('pointer', 'cyan bold')])
    ).ask()
    
    if choice == 'APPROVE':
        return 'APPROVE', None
    elif choice == 'REJECT':
        return 'REJECT', None
    elif choice == 'REASONING':
        return 'REASONING', None
    elif choice == 'EDIT':
        edit_instructions = questionary.text("What changes would you like to make to this plan?").ask()
        return 'EDIT', edit_instructions
    
    return 'REJECT', None

def execute_phase(approved_plan: AgentPlan, all_pois: list):
    """
    Phase 3: Execute the approved logic using Tavily and LLM point queries.
    """
    console.print("\n[bold green]Plan Approved! Commencing execution...[/bold green]\n")
    llm = GroqLLM("llama-3.3-70b-versatile")
    tavily_client = TavilyClient(api_key=config.tavily_api_key)
    
    executor = AgentExecutor(
        llm=llm,
        tavily_client=tavily_client
    )
    poi_dict = {p.poi_id: p for p in all_pois}
    results = executor.execute_plan(approved_plan.groups, poi_dict)
    
    console.print(f"\n[magenta]Processed {len(results)} POIs from independent predictions.[/magenta]")
    return results

def main():
    try:
        # Halt immediately if API keys are missing before spinning up CLI
        config.validate_keys()
        
        print_banner()
        time.sleep(1)
        all_flagged_pois = run_ingest_phase()
        selected_pois = select_pois_phase(all_flagged_pois)
        
        if not selected_pois:
            return
            
        time.sleep(1)
        agent_plan = run_plan_phase(selected_pois)
        
        while True:
            action, edit_instr = user_approval_gate(agent_plan)
            
            if action == 'APPROVE':
                execute_phase(agent_plan, selected_pois)
                break
            elif action == 'REASONING':
                panel = Panel(
                    Text(agent_plan.plan_reasoning, style="italic cyan"),
                    title="[bold cyan]🧠 LLM Reasoning Process[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2)
                )
                console.print(panel)
                print_plan(agent_plan) # Re-print summary table under it
            elif action == 'REJECT':
                console.print("[bold red]Plan rejected by user. Exiting...[/bold red]")
                sys.exit(0)
            elif action == 'EDIT':
                if not edit_instr: continue
                console.print("\n[bold yellow]Sending edit instructions to Agent Planner...[/bold yellow]")
                llm = GroqLLM("llama-3.3-70b-versatile")
                planner = AgentPlanner(llm=llm)
                
                with console.status("[bold yellow]LLM is updating the plan...", spinner="dots"):
                    try:
                        agent_plan = planner.edit_plan(agent_plan, edit_instr)
                    except Exception:
                        console.print("[red]Failed to edit plan. Showing previous plan.[/red]")
                
                print_plan(agent_plan)
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Process interrupted by user. Exiting...[/bold red]")
        sys.exit(0)

if __name__ == "__main__":
    main()
