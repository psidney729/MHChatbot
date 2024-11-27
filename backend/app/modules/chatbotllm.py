# Solution 3 - Chatbot using LLM
from ..config import settings
from ..models import History

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.errors import GraphRecursionError
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from pydantic import BaseModel, Field
from typing import Annotated, Literal, Sequence, TypedDict, List, Dict, Any, Optional
import operator
import functools
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

conn = psycopg2.connect(f"host={settings.POSTGRES_HOST} dbname={settings.POSTGRES_DB} user={settings.POSTGRES_USER} password={settings.POSTGRES_PASSWORD}")
cur = conn.cursor()

# Define llm
llm = ChatOpenAI(model="gpt-4o")
client = OpenAI()

def get_top_k_vectors(query_vector, k: int):
    cur.execute(f"""
        SELECT *, semantic_vector <=> '{query_vector}' as distance
        FROM conversationdb
        ORDER BY semantic_vector <=> '{query_vector}'
    """)
    
    # vectors = cur.fetchall()
    vectors = cur.fetchmany(k)

    return vectors

# Define multi-agent system state


class MedicAgentState(TypedDict):
    sender: str
    messages: Annotated[Sequence[BaseMessage], operator.add]


class ContextResponse(TypedDict):
    context: str = Field(description="Sample question")
    response: str = Field(description="Sample response")


def calculate_embedding(text):
    result = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return result.data[0].embedding


@tool("semantically_similar_questions_answer_fetch")
def semantically_similar_question_answer_fetch(
    state: Annotated[dict, InjectedState]
) -> Annotated[list[ContextResponse], "List of sample (question, response) pairs"]:
    """This tool takes the human input and fetch the most relevant (question, answer) pair"""
    for x in reversed(state['messages']):
        if isinstance(x, HumanMessage):
            query_vector = calculate_embedding(x.content)
            break
    k_similar_vector = get_top_k_vectors(query_vector, 5)
    cotextResponses = [ContextResponse(context=vector[1], response=vector[2]) for vector in k_similar_vector]
    return cotextResponses


class MedicAgent():
    def __init__(self):
        self.llm = llm
        self.workflow = StateGraph(MedicAgentState)

        self.gateAgent = gate_agent(llm=self.llm)
        self.gateNode = functools.partial(
            gate_agent_to_node,
            agent=self.gateAgent,
            name="Gate"
        )

        self.idleAgent = idle_agent(llm=self.llm)
        self.idleNode = functools.partial(
            idle_agent_to_node,
            agent=self.idleAgent,
            name="Idle"
        )

        self.medicAgent = medic_agent(llm=self.llm, tools=tools)
        self.medicNode = functools.partial(
            medic_agent_to_node,
            agent=self.medicAgent,
            name="Medic"
        )

        self.toolNode = ToolNode(tools)

        self.workflow.add_node("Gate", self.gateNode)
        self.workflow.add_node("Idle", self.idleNode)
        self.workflow.add_node("Medic", self.medicNode)
        self.workflow.add_node("call_tool", self.toolNode)

        self.workflow.add_edge(START, "Gate")
        self.workflow.add_conditional_edges(
            "Gate",
            router_gate_node,
            {
                "Medic": "Medic",
                "Idle": "Idle"
            }
        )
        self.workflow.add_conditional_edges(
            "Medic",
            router_medic_node,
            {
                "call_tool": "call_tool",
                "__end__": END
            }
        )
        self.workflow.add_conditional_edges(
            "call_tool",
            lambda state: state["sender"],
            {
                "Medic": "Medic"
            }
        )
        self.workflow.add_edge("Idle", END)

        self.graph = self.workflow.compile()


tools = [
    semantically_similar_question_answer_fetch
]


def gate_agent(llm):
    """Create an agent for gate"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant."
                "User will ask questions."
                "You have to determine whether the question is related to mental health problem or not."
                "If it is not related to mental health problem, you simply respond with 'Not'."
                "If it is related to mental health problem, you respond with 'Medication'."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | llm


def gate_agent_to_node(state, agent, name):
    result = agent.invoke(state)
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "sender": name,
        "messages": [result]
    }


def idle_agent(llm):
    """Create an agent for general questions"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant."
                "User will ask normal questions from user."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | llm


def idle_agent_to_node(state, agent, name):
    result = agent.invoke(state)
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "sender": name,
        "messages": [result]
    }


def medic_agent(llm, tools):
    """Create an agent for medical questions"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant.\n"
                "User will ask questions about mental health.\n"
                "Use the provided tool to progress towards answering the question.\n"
                "If you think you have the answer, prefix your response with FINAL ANSWER.\n"
                "You have access to the following tools: {tool_names}.\n",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join(
        [tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)


def medic_agent_to_node(state, agent, name):
    result = agent.invoke(state)
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "sender": name,
        "messages": [result]
    }


def router_gate_node(state) -> Literal["Medic", "Idle"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content.startswith("Not"):
        return "Idle"
    else:
        return "Medic"


def router_medic_node(state) -> Literal["call_tool", "continue", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    else: 
        return "__end__"
    
def fetch_all_history(user_uuid):
    cur.execute(f"""
        SELECT user_message, llm_message
        FROM history
        WHERE user_uuid = '{user_uuid}'
    """)

    history = cur.fetchall()
    return history


def save_new_message(user_uuid, user_message, llm_message):
    user_message = user_message.replace("'", " ")
    llm_message = llm_message.replace("'", " ")
    cur.execute(f"""
        INSERT INTO history (user_uuid, user_message, llm_message)
        VALUES ('{user_uuid}', '{user_message}', '{llm_message}')
    """)
    conn.commit()
    
def chatbotllm(message, current_user_uuid):
    medic_agent = MedicAgent()
    config = {"recursion_limit": 10, "configurable": {"thread_id": "MVP_TEST"}}

    dbhistory = fetch_all_history(current_user_uuid)

    history = []
    if len(dbhistory) != 0:
        history = []
        for entry in dbhistory:
            history.append(HumanMessage(entry[0]))
            history.append(AIMessage(entry[1]))

    history.append(HumanMessage(content=message))
    try:
        res = medic_agent.graph.invoke(
            {
                "messages": history,
                "sender": "user",
            },
            config,
        )
        last_message = res["messages"][-1].content
        if last_message.startswith("FINAL ANSWER"):
            last_message = last_message[13:]
        if last_message.endswith("FINAL ANSWER"):
            last_message = last_message[:-13]
    except GraphRecursionError:
        last_message = "Sorry, there was an error in this multi-agent system. Please try again."
    history.append(AIMessage(content=last_message))
    save_new_message(current_user_uuid, message, last_message)

    return last_message


