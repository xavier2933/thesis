import os
import random
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug

# --- 0. DEBUG MODE ---
set_debug(False) # Set to False for cleaner logs; we'll print LLM logic manually

# --- 1. Define Structured Output Schema ---
class Decision(BaseModel):
    next_step: str = Field(description="Action to take: 'actor' (retry), 'teleport' (fix), or 'finish'")
    reasoning: str = Field(description="Brief logic for this choice")

# --- 2. Define the State ---
class RobotState(TypedDict):
    grid_map: dict 
    current_target: Optional[tuple]
    last_error: Optional[str]
    history: List[str] # To help LLM see past failures

# --- 3. Initialize Model ---
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# --- 4. Define the Nodes ---

def actor_planner(state: RobotState):
    """The Actor: Finds the next empty slot."""
    for r in range(16):
        for c in range(16):
            if (r, c) not in state['grid_map']:
                return {"current_target": (r, c)}
    return {"current_target": None}

def robot_execution(state: RobotState):
    x, y = state['current_target']
    
    # FORCE AN ERROR at (0, 5) to see the LLM in action
    if x == 0 and y == 5:
        return {"last_error": "Antenna tilted - potential collision"}
    
    # Otherwise, normal random failures
    if random.random() < 0.1:
        return {"last_error": "Alignment Failed"}
    
    new_grid = state['grid_map'].copy()
    new_grid[(x, y)] = "deployed"
    return {"grid_map": new_grid, "last_error": None}

def llm_router(state: RobotState):
    """The LLM Brain: Now it logs EVERY transition."""
    if state["current_target"] is None:
        return "finish"

    structured_llm = llm.with_structured_output(Decision)
    
    # We ask the LLM to decide what to do regardless of error status
    status = "SUCCESS" if not state.get("last_error") else f"ERROR ({state['last_error']})"
    
    prompt = (
        f"Robot Status: {status} at {state['current_target']}. "
        "Grid Progress: {len(state['grid_map'])}/256. "
        "Decide: 'actor' to continue/retry, or 'teleport' if there is an error."
    )
    
    decision = structured_llm.invoke(prompt)
    
    # THIS IS THE LOG YOU WANTED:
    print(f"🤖 [LLM LOG] Status: {status} | Decision: {decision.next_step.upper()} | Reason: {decision.reasoning}")
    
    if state.get("last_error"):
        return decision.next_step
    return "next_antenna"

def critic_node(state: RobotState):
    """Logs progress and decides if LLM intervention is needed."""
    if not state.get("last_error"):
        print(f"✅ Success at {state['current_target']}. Grid size: {len(state['grid_map'])}")
        return {"last_error": None}
    
    # If error exists, we keep it in state for the router
    print(f"⚠️ Robot reported error: {state['last_error']} at {state['current_target']}")
    return {"last_error": state["last_error"]}

def teleport_recovery(state: RobotState):
    """The Recovery: The 'Stochastic Teleport' hack."""
    x, y = state['current_target']
    print(f"🔮 Recovery: Teleporting antenna to {x}, {y}...")
    new_grid = state['grid_map'].copy()
    new_grid[(x, y)] = "teleported"
    return {"grid_map": new_grid, "last_error": None}



# --- 6. Build Graph ---
workflow = StateGraph(RobotState)

workflow.add_node("actor", actor_planner)
workflow.add_node("robot", robot_execution)
workflow.add_node("critic", critic_node)
workflow.add_node("teleport", teleport_recovery)

workflow.add_edge(START, "actor")
workflow.add_edge("actor", "robot")
workflow.add_edge("robot", "critic")

workflow.add_conditional_edges(
    "critic",
    llm_router,
    {
        "actor": "actor",
        "teleport": "teleport",
        "next_antenna": "actor",
        "finish": END
    }
)
workflow.add_edge("teleport", "actor")

app = workflow.compile(checkpointer=MemorySaver())

# --- 7. Run with Safety Fuse ---
config = {"configurable": {"thread_id": "test_run_2"}, "recursion_limit": 100}
initial_state = {"grid_map": {}, "history": [], "last_error": None}

print("--- Starting Mission ---")
try:
    for event in app.stream(initial_state, config):
        pass # The print statements in nodes show us the progress
except Exception as e:
    print(f"\n🛑 MISSION HALTED: {e}")