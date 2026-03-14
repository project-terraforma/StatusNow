import os
import requests
from config import config

def yelp_search(term: str, location: str) -> dict:
    """Search for businesses by keyword and location."""
    if not config.yelp_api_key:
        return {"error": "YELP_API_KEY not configured."}
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {config.yelp_api_key}"}
    params = {"term": term, "location": location, "limit": 2}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        search_data = resp.json()
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
        
    businesses = search_data.get("businesses", [])
    if not businesses:
        return {"search_results": [], "message": "No businesses found."}
        
    enriched = []
    for b in businesses:
        b_id = b.get("id")
        result = {"basic_info": b}
        
        # Fetch details
        try:
            d_resp = requests.get(f"https://api.yelp.com/v3/businesses/{b_id}", headers=headers, timeout=10)
            if d_resp.status_code == 200:
                result["details"] = d_resp.json()
        except Exception:
            pass
            
        # Fetch reviews
        try:
            r_resp = requests.get(f"https://api.yelp.com/v3/businesses/{b_id}/reviews", headers=headers, timeout=10)
            if r_resp.status_code == 200:
                result["reviews"] = r_resp.json().get("reviews", [])
        except Exception:
            pass
            
        enriched.append(result)
        
    return {"bundled_results": enriched}

def yelp_business_details(business_id: str) -> dict:
    """Get rich business data, such as hours of operation and photos."""
    if not config.yelp_api_key:
        return {"error": "YELP_API_KEY not configured."}
    url = f"https://api.yelp.com/v3/businesses/{business_id}"
    headers = {"Authorization": f"Bearer {config.yelp_api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def yelp_reviews(business_id: str) -> dict:
    """Get up to three review excerpts for a business."""
    if not config.yelp_api_key:
        return {"error": "YELP_API_KEY not configured."}
    url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
    headers = {"Authorization": f"Bearer {config.yelp_api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def tavily_extract(tavily_client, urls: list[str]) -> dict:
    """Extract full content from specific URLs using Tavily's extract API."""
    try:
        # Tavily's extract endpoint expects a list of URLs
        return tavily_client.extract(urls=urls)
    except Exception as e:
        return {"error": str(e)}

def execute_tool(tavily_client, tool_name: str, args: dict) -> dict:
    """Dispatcher for the agent's tool requests."""
    if tool_name == "yelp_search":
        return yelp_search(args.get("term", ""), args.get("location", ""))
    elif tool_name == "yelp_business_details":
        return yelp_business_details(args.get("business_id", ""))
    elif tool_name == "yelp_reviews":
        return yelp_reviews(args.get("business_id", ""))
    elif tool_name == "tavily_extract":
        urls = args.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        return tavily_extract(tavily_client, urls)
    elif tool_name == "tavily_search":
        from tavily import TavilyClient
        try:
            return tavily_client.search(query=args.get("query", ""), max_results=3)
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": f"Unknown tool: {tool_name}"}
