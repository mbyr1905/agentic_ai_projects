from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

web_search_agent = Agent(
    name='Web Search Agent',
    role='Search the web for latest info',
    model = Groq(id='llama-3.3-70b-versatile'),
    tools=[DuckDuckGo()],
    instructions=['Always include sources'],
    show_tool_calls = True,
    markdown=True
)

finance_agent = Agent(
    name='Finance AI Agent',
    model = Groq(id='llama-3.3-70b-versatile'),
    tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
            key_financial_ratios=True
        )
    ],
    instructions=['Use tables and graphs to diaply data'],
    show_tool_calls = True,
    markdown=True
)

app=Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)