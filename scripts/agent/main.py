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
    
    panel = Panel(
        "[bold white]How many POIs would you like to process?[/bold white]\n"
        "[dim]Options:[/dim]\n"
        "  [[bold cyan]1[/bold cyan]] Pick 1 random POI for deep analysis\n"
        "  [[bold cyan]2[/bold cyan]] Pick 10 random POIs for broad analysis\n"
        "  [[bold cyan]3[/bold cyan]] Pick custom POIs (Paginate & select)",
        title="[bold cyan]Selection Mode[/bold cyan]",
        border_style="cyan"
    )
    console.print(panel)
    
    while True:
        choice = console.input("\n[bold blue]Selection>[/bold blue] ").strip()
        if choice == '1':
            selected = random.sample(all_flagged_pois, min(1, len(all_flagged_pois)))
            console.print(f"[green]Selected {len(selected)} random POI(s).[/green]")
            return selected
        elif choice == '2':
            num = min(10, len(all_flagged_pois))
            selected = random.sample(all_flagged_pois, num)
            console.print(f"[green]Selected {num} random POIs.[/green]")
            return selected
        elif choice == '3':
            return custom_poi_selection(all_flagged_pois)
        else:
            console.print("[red]Invalid choice. Please enter 1, 2, or 3.[/red]")

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
        
        table = Table(title=f"Custom Selection - Page {page+1}/{total_pages}", show_header=True)
        table.add_column("Idx", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Category", style="green")
        table.add_column("Selected", style="yellow")
        
        for i, poi in enumerate(current_page_pois):
            global_idx = start_idx + i
            is_selected = "✓" if any(p.poi_id == poi.poi_id for p in selected_pois) else ""
            table.add_row(str(global_idx), str(poi.name), str(poi.category), is_selected)
            
        console.print(table)
        console.print(f"[dim]Selected so far: {len(selected_pois)}[/dim]")
        
        prompt = "[N]ext page, [P]revious page, [D]one. Or enter comma-separated indices to add (e.g., 0, 2, 5)"
        choice = console.input(f"\n[bold blue]User Input>[/bold blue] {prompt}\n> ").strip().lower()
        
        if choice == 'n':
            if page < total_pages - 1:
                page += 1
            else:
                console.print("[yellow]Already on the last page.[/yellow]")
        elif choice == 'p':
            if page > 0:
                page -= 1
            else:
                console.print("[yellow]Already on the first page.[/yellow]")
        elif choice == 'd':
            if not selected_pois:
                console.print("[yellow]No POIs selected. Defaulting to 1 random.[/yellow]")
                return random.sample(all_flagged_pois, min(1, len(all_flagged_pois)))
            return selected_pois
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                for idx in indices:
                    if 0 <= idx < len(all_flagged_pois):
                        poi_to_add = all_flagged_pois[idx]
                        if not any(p.poi_id == poi_to_add.poi_id for p in selected_pois):
                            selected_pois.append(poi_to_add)
                            console.print(f"[green]Added '{poi_to_add.name}'[/green]")
                        else:
                            console.print(f"[yellow]'{poi_to_add.name}' is already selected.[/yellow]")
                    else:
                        console.print(f"[red]Index {idx} out of range.[/red]")
            except ValueError:
                console.print("[red]Invalid input. Use N, P, D, or comma-separated numbers.[/red]")

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
    
    panel = Panel(
        "[bold white]Please review the suggested plan above.[/bold white]\n"
        "[dim]Options:[/dim]\n"
        "  [[bold green]A[/bold green]] Approve plan and begin execution\n"
        "  [[bold yellow]E[/bold yellow]] Edit strategies or drop POIs\n"
        "  [[bold red]R[/bold red]] Reject plan and exit",
        title="[bold red]Action Required[/bold red]",
        border_style="red"
    )
    console.print(panel)
    
    while True:
        choice = console.input("\n[bold blue]User Input>[/bold blue] ").strip().upper()
        if choice == 'A':
            return 'APPROVE', None
        elif choice == 'R':
            return 'REJECT', None
        elif choice == 'E':
            edit_instructions = console.input("\n[bold yellow]What changes would you like to make to this plan?[/bold yellow]\n> ").strip()
            return 'EDIT', edit_instructions
        else:
            console.print("[red]Invalid choice. Please select A, E, or R.[/red]")

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
            elif action == 'REJECT':
                console.print("[bold red]Plan rejected by user. Exiting...[/bold red]")
                sys.exit(0)
            elif action == 'EDIT':
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
