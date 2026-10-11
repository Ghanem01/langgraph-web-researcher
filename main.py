import os  # Lets Python read your API key from the environment.
from typing import Annotated, TypedDict  # Lets us describe the data that travels through the graph.

from ddgs import DDGS # Provides the web-search client.
from langchain_core.tools import tool # Marks a Python function as a tool the model can use.

from dotenv import load_dotenv  # Loads settings from your local .env file.
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage  # ToolMessage represents a tool's result.from langgraph.graph.message import add_messages  # Tells LangGraph how to add new messages to the state.
from langgraph.prebuilt import ToolNode  # Adds a graph node that executes requested tools.
from langchain_openai import ChatOpenAI  # Connects LangChain to OpenRouter's compatible API.
from langgraph.graph import END, START, StateGraph  # Provides the building blocks for a workflow graph.
from langgraph.graph.message import add_messages  # Tells LangGraph to append new messages to the message history.

class AgentState(TypedDict):  # Describes the shared data that moves between graph steps.
    messages: Annotated[list[AnyMessage], add_messages]  # Stores conversation messages and appends new ones.


load_dotenv()  # Loads values from .env into the environment.

api_key = os.getenv("OPENROUTER_API_KEY")  # Reads the key without putting it in the code.
if not api_key:  # Checks that the key was loaded.
    raise SystemExit("OPENROUTER_API_KEY is missing from .env")  # Stops with a clear message if it is missing.


model = ChatOpenAI(  # Creates the chat model that our graph step will call.
    model="nvidia/nemotron-3-ultra-550b-a55b:free",  # Selects the free model you tested.
    base_url="https://openrouter.ai/api/v1",  # Sends requests to OpenRouter.
    api_key=api_key,  # Supplies the key loaded from .env.
)


@tool  # Wraps this function as a tool that can be given to a chat model.
def web_search(query: str) -> str:  # Defines a search tool that accepts a text query.
    """Search the web and return a few result titles, URLs, and snippets."""  # Tells the model what this tool does.
    with DDGS() as search_client:  # Opens the search client and closes it when finished.
        results = search_client.text(query, max_results=5)  # Searches the web for up to five results.

    if not results:  # Checks whether the search returned any results.
        return "No search results were found."  # Gives the model a clear response when there are no results.

    return "\n\n".join(  # Combines the results into one text response for the model.
        f"Title: {result.get('title', 'Untitled')}\n"  # Includes the page title when available.
        f"URL: {result.get('href', 'No URL')}\n"  # Includes the result URL when available.
        f"Snippet: {result.get('body', 'No snippet')}"  # Includes the search-result summary when available.
        for result in results  # Formats each result returned by the search.
    )

def answer_question(state: AgentState) -> dict[str, list[AIMessage]]:  # Calls the model using the current graph messages.
    model_with_tools = model.bind_tools([web_search])  # Makes the web-search tool available to the model.
    response = model_with_tools.invoke(  # Sends the instruction and conversation to the model.
        [
            SystemMessage(  # Tells the model when to search and how to handle private facts.
                content=(
                    "Be helpful and accurate. "
                    "Use web_search for current facts, such as today's sports scores. "
                    "If you are unsure, say so instead of inventing an answer. "
                    "Do not use web search to guess private facts about the user."
                )
            ),
            *state["messages"],  # Adds the user's question and any tool results to the request.
        ]
    )
    return {"messages": [response]}  # Adds the model's answer or tool request to the graph state.


def route_after_answer(state: AgentState) -> str:  # Chooses the next graph step based on the model response.
    last_message = state["messages"][-1]  # Gets the newest message in the graph state.
    if isinstance(last_message, AIMessage) and last_message.tool_calls:  # Checks whether the model requested a tool.
        return "tools"  # Routes to the node that runs web_search.
    return END  # Finishes when the model has answered without requesting a tool.


workflow = StateGraph(AgentState)  # Creates a graph that shares the messages state.
workflow.add_node("answer_question", answer_question)  # Adds the model-calling step.
workflow.add_node("tools", ToolNode([web_search]))  # Adds a step that can execute web_search.
workflow.add_edge(START, "answer_question")  # Starts by asking the model what to do.
workflow.add_conditional_edges("answer_question", route_after_answer)  # Searches if requested; otherwise ends.
workflow.add_edge("tools", "answer_question")  # Sends search results back to the model for a final answer.

app = workflow.compile()  # Compiles the workflow so it can run.

question = input("Ask a question: ")  # Reads the question you type.
result = app.invoke({"messages": [HumanMessage(content=question)]})  # Starts the graph with your question.
for message in result["messages"]:  # Inspects every message produced during this graph run.
    if isinstance(message, AIMessage) and message.tool_calls:  # Finds any search request made by the model.
        for tool_call in message.tool_calls:  # Checks each tool request in case there is more than one.
            print(f"\nSearch query: {tool_call['args'].get('query')}")  # Shows the query the model actually searched for.
    elif isinstance(message, ToolMessage):  # Finds the result returned by the search tool.
        print(f"\nSearch results used by agent:\n{message.content}")  # Shows the raw results the model received.

print(f"\nAgent > {result['messages'][-1].content}")  # Shows the model's final answer after the trace.