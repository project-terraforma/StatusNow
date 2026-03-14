import json
import sys
from pathlib import Path
from typing import List, Dict

from llm.interface import LLMInterface
from schemas import POIRecord, AgentPlan, PlanGroup
from rich.console import Console

PLANNER_SYSTEM_PROMPT = """
You are the Lead OSINT Strategist for the StatusNow V6 POI Pipeline.
Your objective is to analyze a batch of Points of Interest (POIs) that a machine learning model struggled to classify as Open or Closed, and devise an ultra-efficient web search strategy for them.

You have a strict budget of 4,000 Tavily search credits. 
You must output a structured JSON plan organizing these POIs into strategic groups.

Available Strategies:
1. `web_search_name_address`: Standard check. Use for generic local businesses.
2. `web_search_business_status`: Aggressive closure check. Use for POIs where the model is confident it's closed but we need proof.
3. `web_search_category_specific`: Deep search. Use for high-churn categories (e.g., restaurants need menu checks, hotels need booking checks).
4. `skip`: Use for places that cannot be verified online (e.g., "Park Bench", "ATM", "Redbox"). These cost 0 credits.

For every group you create, you MUST provide an ordered list of `query_templates` (fallbacks).
The agent will run the first template, evaluate confidence, and ONLY run the second template if the first fails.

Query Template Variables:
You can use `{name}`, `{address}`, and `{category}` in your string templates.

Cost Guidelines:
- A single Tavily API call costs 1 credit. 
- Try to set `max_results` to 1 or 2 for initial broad searches to save credits and context window. 
- Deep category searches might require `max_results=3`.
"""

class AgentPlanner:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def generate_plan(self, flagged_pois: List[Dict]) -> AgentPlan:
        """
        Sends the batch of POIs to Gemini requesting a strict JSON AgentPlan back.
        """
        # Truncate descriptions to save context window if necessary
        formatted_pois = []
        for p in flagged_pois:
            formatted_pois.append({
                "poi_id": p.poi_id,
                "name": p.name,
                "category": p.category,
                "address": p.address
            })
            
        prompt = f"Analyze the following {len(formatted_pois)} POIs and generate an execution plan:\n{json.dumps(formatted_pois, indent=2)}"
        
        try:
            # We enforce the AgentPlan Pydantic schema using Gemini's Structured Outputs
            return self.llm.generate_structured_output(
                PLANNER_SYSTEM_PROMPT + "\n\n" + prompt,
                AgentPlan
            )
        except Exception as e:
            print(f"Error generating plan: {e}")
            raise

    def edit_plan(self, existing_plan: AgentPlan, user_instructions: str) -> AgentPlan:
        """
        Takes an existing plan and user feedback, and generates a modified plan using Gemini.
        """
        # We need to dump the model strictly for context. We can assume `existing_plan` is an AgentPlan
        try:
            plan_json = existing_plan.model_dump_json(indent=2)
        except AttributeError:
            # fallback if it's already dict/json or something else
            plan_json = str(existing_plan)
            
        prompt = (
            f"Here is the existing plan:\n{plan_json}\n\n"
            f"The user requested the following modifications to this plan: '{user_instructions}'.\n"
            f"Please output the completely updated JSON plan adhering to the exact same schema. "
            f"Do not lose any POIs unless the user explicitly requested it."
        )
        
        try:
            return self.llm.generate_structured_output(
                PLANNER_SYSTEM_PROMPT + "\n\n" + prompt,
                AgentPlan
            )
        except Exception as e:
            print(f"Error editing plan: {e}")
            raise
