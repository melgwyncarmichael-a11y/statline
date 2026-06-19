import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

DB_PATH = os.path.join(os.path.dirname(__file__), "database.sqlite")

db = SQLDatabase.from_uri(
    f"sqlite:///{DB_PATH}",
    include_tables=["Match", "Player", "Team", "League", "Player_Attributes"]
)

llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0
)

agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    handle_parsing_errors=True
)

result = agent.invoke("Which team scored the most goals across all seasons?")
print("\n--- ANSWER ---")
print(result["output"])
