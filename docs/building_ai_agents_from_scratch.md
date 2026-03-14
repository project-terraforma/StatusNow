# Building My First AI Agent From Scratch (And the Hilarious Bugs I Found)

If you've spent any time reading about AI lately, you've probably heard the term "Agent" thrown around everywhere. Unlike standard chatbots that just talk back to you, an Agent is given a goal, a set of tools (like web search or database access), and the autonomy to think, plan, and execute actions in a loop until it reaches a conclusion.

Instead of reaching for heavy frameworks like LangChain or AutoGen, I decided to build my first AI Agent entirely from scratch in Python to classify whether businesses (Points of Interest, or POIs) were Open or Permanently Closed. I wanted to see exactly how the "magic" worked under the hood.

It turns out the magic isn't actually magic—it's just a while loop, some structured JSON schemas, and a whole lot of prompt engineering. But along the way, I ran into some hilarious and frustrating hallucinations that taught me exactly how LLMs "think" when given agency.

Here is what I learned, the bugs I encountered, and how I fixed them.

## The Goal

The mission for the agent was simple:

1. Look at a list of low-confidence business locations.
2. Formulate a search plan for each one.
3. Iteratively use tools (Tavily for web searches, Yelp API for business details).
4. Decide with >80% confidence if the business is OPEN or CLOSED.

I wired up Google's Gemini-2.5-Flash (and later Llama-3 via Groq) to a while loop that kept asking the LLM to output a strictly typed Pydantic object containing its reasoning, a requested_tool, and a final predicted_label if it was confident enough to stop.

## Bug #1: The Infinite Tool-Calling Loop (Amnesia)

The single biggest issue I ran into was that my agent would keep calling the exact same tool query over and over again.

Executing `yelp_search`... LLM requested tool: `yelp_search` with args `{'term': 'Mobile Semi Truck Mechanic Medley', 'location': 'Medley, FL'}`
Executing `yelp_search`... LLM requested tool: `yelp_search` with args `{'term': 'Mobile Semi Truck Mechanic Medley', 'location': 'Medley, FL'}`

It was stuck in a desperate, infinite loop. Why? Because LLMs are inherently stateless.

In my initial `executor.py` loop, I was taking the output of the requested tool (like a JSON blob of Yelp search results) and appending it to the conversation_context list. But I wasn't appending the LLM's own prior actions.

When the LLM generated its next response, it saw the Yelp data sitting in its prompt context, but it had no memory of the fact that it was the one who just asked for it. It thought, "I need Yelp data. I should call the Yelp search tool!" over and over again.

### The Fix

I had to explicitly re-inject the LLM's own historical decisions alongside the tool outputs back into the rolling prompt buffer. I updated the script to format the history exactly like this before throwing it back into the system prompt:

```text
--- Turn 1 History ---
Your Reasoning: The Yelp search results do not include the POI, which makes it difficult to determine its status.
You Requested Tool: 'yelp_search' with args {'term': 'Mobile Semi', 'location': 'Medley, FL'}
Tool Output Received:
{"businesses": [], "total": 0}
```

Once I explicitly told the LLM what it had just done, the amnesia vanished. It instantly realized it had already exhausted that search query and naturally pivoted to trying a broader Google search instead!

## Bug #2: The Desperate Empty Argument ({})

Another funny behavior emerged when the agent wanted to look up a business's Yelp details. The tool `yelp_business_details` required a specific `business_id` (an alias like `my-business-miami-2`) that the agent was supposed to find from a prior `yelp_search`.

But when the initial `yelp_search` returned zero results, the agent still really wanted to check the business details anyway. Since it didn't have an ID, it just blindly fired the tool with empty arguments: `yelp_business_details` with args `{}`.

The API understandably threw a 404 error, and the LLM looped again.

### The Fix: Prompt Engineering with a Stick

I initially thought I needed complex Python validation logic to block bad tool calls. It turns out, you just need to yell at the LLM in the system prompt.

I updated the tool instructions string from this:
`- 'yelp_business_details': args {"business_id": "..."} (Fetch Yelp listing status. Requires a Yelp business_id)`

To this critically strict instruction:
`- 'yelp_business_details': args {"business_id": "..."} (Fetch Yelp listing status. CRITICAL: You CANNOT use this unless you already know the exact Yelp business_id alias from a previous yelp_search)`

Once the LLM read the word **CRITICAL** and explicitly understood the dependency chain, the empty argument calls stopped entirely. It fell back to generating a final prediction rather than guessing tool inputs.

## Part 2: Async Agent for Scalability and Batching

As the agent got faster, it hit a new wall: API Rate Limits. Groq's free tier is generous, but as soon as I started running the agent in an asynchronous loop, I was hitting `429: Too Many Requests` status codes within seconds. Each POI was triggering multiple API calls in quick succession, overwhelming the "Tokens Per Minute" bucket.

### The Fix: Single-Prompt Batching

Instead of calling the API for every individual step of every individual business, I refactored the pipeline to use **Single-Prompt Batching**.

I transformed the "Research Worker" to gather evidence for 5 POIs at once. Then, instead of 5 separate inference calls, the agent sends one high-context prompt containing a JSON array of all 5 research bundles.

```json
{
  "batch": [
    {"name": "POI 1", "research": "..."},
    {"name": "POI 2", "research": "..."},
    ...
  ]
}
```

By requesting a **Structured Output** that returns a list of 5 predictions in one go, I reduced the API chatter by 80%. Combined with an **Exponential Backoff** (waiting 30-60s on a 429 error), the agent now glides through thousands of records without breaking a sweat.

## Takeaways from Building From Scratch

Building an agent natively instead of using LangChain was the best decision I could have made for my first project.

When you control the literal strings being concatenated in the while loop, there is zero "magic." You have total visibility into why the model is hallucinating, and fixing it is usually as simple as adding a specific instruction to the prompt context.

I ended up wrapping the whole thing in a beautiful interactive terminal UI using `questionary` and `rich`, and implementing Pydantic structured outputs so I could hot-swap between Groq's insanely fast open-source Llama-3 models and Gemini API.

If you are looking to build your first AI agent: **Don't use a framework right away.** Build the while-loop yourself. You will learn exponentially more about how LLMs actually "think"!
