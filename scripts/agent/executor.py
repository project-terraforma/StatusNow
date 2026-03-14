import os
from typing import List
from schemas import POIRecord, AgentPrediction
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from llm.interface import LLMInterface

console = Console()

class AgentExecutor:
    def __init__(self, llm: LLMInterface, tavily_client=None):
        self.llm = llm
        self.tavily = tavily_client

    def execute_plan(self, approved_groups: list, all_pois: dict) -> list:
        """
        The main inference extraction loop. Iterates through the approved plan,
        running web searches and utilizing Gemini for direct prediction (Open/Closed).
        """
        results = []
        
        # Calculate total pois in the approved groups
        total_tasks = sum(len(group.poi_ids) for group in approved_groups if group.strategy != "skip")
        
        console.print(f"\n[bold green]Executing Approved Plan[/bold green]: Enacting Search & Direct Prediction")
        
        import json
        from agent_tools import execute_tool

        poi_count = 0
        for group in approved_groups:
            if group.strategy == "skip":
                continue
                
            for poi_id in group.poi_ids:
                poi = all_pois.get(poi_id)
                if not poi:
                    continue
                    
                poi_count += 1
                console.rule(f"[bold cyan]Processng POI {poi_count}/{total_tasks}: {poi.name}[/bold cyan]")
                console.print(f"Location: {poi.address}")
                console.print(f"Category: {poi.category}\n")
                
                best_prediction = None
                used_queries = []
                all_sources = []

                # Bootstrap step: Initial Search from plan
                initial_query = group.query_templates[0].format(
                    name=poi.name, address=poi.address or "", category=poi.category or ""
                )
                used_queries.append(initial_query)
                console.print(f"[bold yellow]Initial Search:[/bold yellow] Tavily -> [italic]\"{initial_query}\"[/italic]")
                
                try:
                    search_data = self.tavily.search(query=initial_query, max_results=group.max_results)
                    results_urls = [res.get('url') for res in search_data.get('results', [])]
                    all_sources.extend(results_urls)
                    console.print(f"[green]Found {len(results_urls)} initial sources.[/green]")
                    for url in results_urls:
                        console.print(f"  - [dim]{url}[/dim]")
                except Exception as e:
                    console.print(f"[red]Initial search failed:[/red] {e}")
                    search_data = {"error": str(e)}

                conversation_context = [
                    f"POI info: Name: {poi.name}, Address: {poi.address}, Category: {poi.category}",
                    f"Initial Web Search Result for '{initial_query}':\n{json.dumps(search_data)}"
                ]
                
                loop_count = 0
                max_loops = 5
                while loop_count < max_loops:
                    loop_count += 1
                    console.print(f"\n[bold yellow]Consulting LLM (Turn {loop_count})...[/bold yellow]")
                    
                    tool_instructions = """
Available tools for deep research:
- 'yelp_search': args {"term": "...", "location": "..."}
- 'yelp_business_details': args {"business_id": "..."} (Fetch Yelp listing status. CRITICAL: You CANNOT use this unless you already know the exact Yelp business_id alias from a previous yelp_search)
- 'yelp_reviews': args {"business_id": "..."} (Check latest customer reviews. CRITICAL: You CANNOT use this unless you already know the exact Yelp business_id alias)
- 'tavily_extract': args {"urls": ["http..."]} (Crawl or scrape the full text of a specific webpage)
- 'tavily_search': args {"query": "..."} (Run another Google search with different keywords)

Given this accumulated data, determine if POI is Open or Closed.
If you need more info to reach 0.8+ confidence, request a tool by setting `requested_tool` and `requested_tool_args`.
If you are confident, output the prediction (1 or 0) and set requested_tool to 'none'.
"""
                    prompt = "\n\n---\n\n".join(conversation_context) + "\n" + tool_instructions
                    
                    try:
                        agent_prediction = self.llm.generate_structured_output(prompt, AgentPrediction)
                    except Exception as e:
                        console.print(f"[red]LLM inference failed:[/red] {e}")
                        break
                        
                    best_prediction = agent_prediction
                    console.print(f"[bold white]LLM Reasoning:[/bold white] {best_prediction.reasoning}")
                    
                    if best_prediction.requested_tool and str(best_prediction.requested_tool).lower() != "none" and str(best_prediction.requested_tool).strip() != "":
                        tool_name = best_prediction.requested_tool
                        tool_args = best_prediction.requested_tool_args or {}
                        
                        console.print(f"[magenta]LLM requested tool:[/magenta] {tool_name} with args {tool_args}")
                        
                        # Stop and ask user before calling the tool
                        choice = console.input(f"[bold blue]User Input>[/bold blue] Press Enter to allow tool execution and loop again, or 'S' to Stop and force prediction: ").strip().lower()
                        if choice == 's':
                            console.print(f"[bold yellow]User force-stopped inference loop for {poi.name}.[/bold yellow]\n")
                            break
                            
                        # Execute the tool
                        console.print(f"[cyan]Executing {tool_name}...[/cyan]")
                        tool_res = execute_tool(self.tavily, tool_name, tool_args)
                        
                        # Format the result to avoid token overflow
                        tool_res_str = json.dumps(tool_res)
                        if len(tool_res_str) > 15000:
                            tool_res_str = tool_res_str[:15000] + "... (truncated)"
                            
                        # Feed the agent's action and tool output back into context for the next turn
                        turn_history = (
                            f"--- Turn {loop_count} History ---\n"
                            f"Your Reasoning: {best_prediction.reasoning}\n"
                            f"You Requested Tool: '{tool_name}' with args {tool_args}\n"
                            f"Tool Output Received:\n{tool_res_str}"
                        )
                        conversation_context.append(turn_history)
                    else:
                        # Agent has reached a final conclusion
                        status_color = "bold green" if best_prediction.predicted_label == 1 else "bold red"
                        status_text = "OPEN" if best_prediction.predicted_label == 1 else "CLOSED"
                        
                        console.print(f"\n[bold white]Prediction:[/bold white] [{status_color}]{status_text}[/{status_color}] (Confidence: {best_prediction.confidence:.2f})")
                        
                        if best_prediction.confidence >= 0.8:
                            console.print(f"\n[bold green]✓ High confidence match reached. Stopping searches for {poi.name}.[/bold green]\n")
                        else:
                            console.print("\n[bold yellow]⚠️ Confidence is below 0.8, but Gemini didn't request any more tools.[/bold yellow]\n")
                        break
                        
                if best_prediction is not None and str(best_prediction.requested_tool).lower() != "none" and str(best_prediction.requested_tool).strip() != "":
                    if loop_count >= max_loops:
                        console.print(f"\n[red]Max loops ({max_loops}) reached for {poi.name}. Forcing final prediction...[/red]")
                    else:
                        console.print(f"[bold yellow]Forcing final prediction based on user stop...[/bold yellow]")
                        
                    force_prompt = prompt + "\n\nCRITICAL: The research loop was stopped. You MUST provide your final best-guess prediction now based on the accumulated context. Set `predicted_label` to 1 or 0, and `requested_tool` to 'none'."
                    console.print(f"[bold yellow]Consulting LLM for forced final prediction...[/bold yellow]")
                    try:
                        best_prediction = self.llm.generate_structured_output(force_prompt, AgentPrediction)
                        status_color = "bold green" if best_prediction.predicted_label == 1 else "bold red"
                        status_text = "OPEN" if best_prediction.predicted_label == 1 else "CLOSED"
                        console.print(f"[bold white]LLM Reasoning:[/bold white] {best_prediction.reasoning}")
                        console.print(f"\n[bold white]Forced Prediction:[/bold white] [{status_color}]{status_text}[/{status_color}] (Confidence: {best_prediction.confidence:.2f})\n")
                    except Exception as e:
                        console.print(f"[red]Failed to generate forced prediction:[/red] {e}")

                if best_prediction is None:
                    console.print(f"[red]Failed to extract prediction for {poi.name}[/red]")
                    continue
                    
                # Store Result
                results.append({
                    "poi_id": poi.poi_id,
                    "prediction": best_prediction,
                    "used_queries": used_queries,
                    "sources": list(set(all_sources))
                })
                
                # Checkpoint at end of POI
                if poi_count < total_tasks:
                   console.input("[dim]Press Enter to proceed to the next POI...[/dim]")

        console.print("\n[bold green]Execution Phase Complete.[/bold green] All predictions generated and captured.")
        return results
