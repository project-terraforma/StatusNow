import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime

# --- Ingestion Schemas ---

class POIRecord(BaseModel):
    poi_id: str
    name: str
    category: str
    original_label: int  # 1 = Open, 0 = Closed
    original_confidence: float
    address: Optional[str] = None
    existing_digital_signals: Dict[str, float] = Field(default_factory=dict)

# --- Planning Phase Schemas ---

class PlanGroup(BaseModel):
    group_id: str
    strategy: Literal[
        "web_search_name_address", 
        "web_search_business_status", 
        "web_search_category_specific", 
        "skip"
    ]
    query_templates: List[str] = Field(description="An ordered list of template strings for Tavily search. If the first yields low confidence, it falls back to the next. Can be just 1. E.g., ['is {name} at {address} permanently closed', '{name} {category} {address} official website']")
    poi_ids: List[str]
    estimated_credits_per_poi: int
    max_results: int = Field(3, description="Number of search results to fetch per query. Can be configured to 1.")

class AgentPlan(BaseModel):
    plan_id: str
    created_at: str
    total_pois: int
    total_estimated_credits: int
    groups: List[PlanGroup]

# --- Execution Phase Schemas ---

class AgentPrediction(BaseModel):
    predicted_label: Optional[int] = Field(None, description="1 if the POI is Open, 0 if it is Closed. None if you need more info.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0. If below 0.8, request a tool!")
    reasoning: str = Field(description="Brief explanation of your reasoning or why you are calling a tool.")
    requested_tool: Optional[Literal["yelp_search", "yelp_business_details", "yelp_reviews", "tavily_extract", "tavily_search", "none"]] = Field("none", description="Tool to call if confidence is low.")
    requested_tool_args: Optional[Dict[str, str]] = Field(default_factory=dict, description="Arguments for the tool (e.g. {'term': 'Pizza', 'location': 'NY'})")

class ResolutionStatus(BaseModel):
    status: Literal[
        "enriched_and_updated", 
        "enriched_no_change", 
        "unenriched_api_error", 
        "skipped_by_user"
    ]

# --- Output Schemas ---

class EnrichmentResult(BaseModel):
    poi_id: str
    original_label: int
    original_confidence: float
    enriched_label: Optional[int] = None
    enriched_confidence: Optional[float] = None
    confidence_delta: Optional[float] = None
    resolution_status: str
    search_queries: List[str]
    sources: List[str]
    agent_prediction: AgentPrediction
