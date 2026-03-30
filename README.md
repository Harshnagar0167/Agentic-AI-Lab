# Autonomous Research Agent using LangChain

## Objective
This project builds an AI agent that automatically researches a topic and generates a structured report.

## Features
- Accepts a research topic as input
- Searches the web using Tavily
- Uses Wikipedia as a knowledge tool
- Analyzes and organizes information
- Generates a final detailed report

## Tech Stack
- Python
- LangChain
- Groq API
- Tavily Search API
- Wikipedia API Wrapper

## Tools Used
1. Web Search Tool
2. Knowledge Tool (Wikipedia)

## Agent Type
ReAct-like autonomous agent using LangChain `create_agent`

## How to Run
Create a .env file:

GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key

```bash
pip install -r requirements.txt
python research_agent.py