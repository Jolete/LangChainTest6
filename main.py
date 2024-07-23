from modules.environment.environment_utilities import (
    load_environment_variables,
    verify_environment_variables,
)
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import neo4j_graph
from uuid import uuid4


# Main program
try:
   
    #region Load environtment

    # Load environment variables using the utility
    env_vars = load_environment_variables()
    
    # Verify the environment variables
    if not verify_environment_variables(env_vars):
        raise ValueError("Some environment variables are missing!")

    llm = OpenAI(
        openai_api_key=env_vars["OPEN_AI_SECRET_KEY"],
        model="gpt-3.5-turbo-instruct",
        temperature=0)
  
    #endregion

    #region neo4j db 
    
    graph = neo4j_graph.Neo4jGraph(
        url=env_vars["NEO4J_URI"],
        username=env_vars["NEO4J_USERNAME"],
        password=env_vars["NEO4J_PASSWORD"]
    )

    #endregion

    #region Session & Prompt & Memory & Tools
    SESSION_ID = str(uuid4())
    print(f"Session ID: {SESSION_ID}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a movie expert. You find movies from a genre or plot.",
            ),
            ("human", "{input}"),
        ]
    )

    movie_chat = prompt | llm | StrOutputParser()

    youtube = YouTubeSearchTool()

    def get_memory(session_id):
        return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

    def call_trailer_search(input):
        input = input.replace(",", " ")
        return youtube.run(input)

    tools = [
        Tool.from_function(
            name="Movie Chat",
            description="For when you need to chat about movies. The question will be a string. Return a string.",
            func=movie_chat.invoke,
        ),
        Tool.from_function(
            name="Movie Trailer Search",
            description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
            func=call_trailer_search,
        ),
    ]
    #endregion

    #region Agent configuration

    agent_prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    chat_agent = RunnableWithMessageHistory(
        agent_executor,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    #endregion 

    while True:
        q = input("> ")

        response = chat_agent.invoke(
            {
                "input": q
            },
            {"configurable": {"session_id": SESSION_ID}},
        )
        
        print(response["output"])

except Exception as e:
    print(f"An unexpected error occurred: {e}")