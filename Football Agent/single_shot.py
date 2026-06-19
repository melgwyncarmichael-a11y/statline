import os, re
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

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

schema = db.get_table_info()

def ask(question: str):
    prompt = f"""You are a SQL expert. Given the schema below, write a SQLite query to answer the question.

Schema:
{schema}

Question: {question}

Reply in this exact format:
SQL:
```sql
<your query here>
```
Answer: <one sentence explaining what the query will return>"""

    response = llm.invoke(prompt).content

    # parse SQL from response
    match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
    if not match:
        print("Could not parse SQL from response.")
        print(response)
        return

    sql = match.group(1).strip()
    answer_line = re.search(r"Answer:\s*(.+)", response)
    explanation = answer_line.group(1).strip() if answer_line else ""

    print(f"\nQuestion: {question}")
    print(f"\nSQL:\n{sql}")

    results = db.run(sql)
    print(f"\nResults: {results}")
    print(f"\nAnswer: {explanation}")

ask("Which team scored the most goals across all seasons?")
