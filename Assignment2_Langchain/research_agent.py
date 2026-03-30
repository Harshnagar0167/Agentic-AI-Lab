import os
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient
from pathlib import Path


# -----------------------------
# Environment Variable Check
# -----------------------------
def check_env():
    required = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variable(s): {', '.join(missing)}\n"
            f"Please set them before running the program."
        )


# -----------------------------
# Tool 1: Web Search via Tavily
# -----------------------------
@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and relevant information on a topic.
    Returns summarized search results with titles and URLs.
    """
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
        include_raw_content=False,
    )

    parts = []

    answer = response.get("answer")
    if answer:
        parts.append(f"Summary Answer:\n{answer}\n")

    results = response.get("results", [])
    if results:
        parts.append("Top Web Results:")
        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content")
            parts.append(f"{i}. {title}\nURL: {url}\nContent: {content}\n")

    return "\n".join(parts) if parts else "No web results found."


# -----------------------------
# Tool 2: Wikipedia Tool
# -----------------------------
wiki_api = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=3000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)


# -----------------------------
# Report Formatter
# -----------------------------
def build_final_report(topic: str, research_notes: str) -> str:
    """
    Takes topic + research notes and asks the LLM to create
    the final structured assignment report.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = f"""
You are an expert research report writer.

Create a detailed, well-structured report on the topic:
"{topic}"

Use the research notes below:
{research_notes}

The output MUST follow this exact format:

1. Cover Page
   - Assignment Name
   - Student Name
   - Course Name
   - Submission Date
   - Topic

2. Title

3. Introduction

4. Key Findings
   - Use numbered or bulleted points
   - Explain each point clearly

5. Challenges

6. Future Scope

7. Conclusion

Rules:
- Write in clear academic language
- Make the report detailed but readable
- Do not invent facts not supported by the notes
- Keep the structure clean
"""

    response = llm.invoke(prompt)
    return response.content


# -----------------------------
# Research Agent Runner
# -----------------------------
def run_research_agent(topic: str) -> str:
    """
    Runs the agent with two tools:
    1) web_search
    2) wikipedia_tool
    Then uses the gathered output to generate the final report.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    agent = create_agent(
        model=llm,
        tools=[web_search, wikipedia_tool],
        system_prompt=(
            "You are an autonomous research agent. "
            "Your job is to research the user's topic using the available tools, "
            "collect useful facts, compare information from multiple sources, "
            "and produce organized research notes."
        ),
    )

    user_prompt = f"""
Research the topic: {topic}

Instructions:
1. Use the web_search tool to gather recent information.
2. Use the Wikipedia tool to gather background/encyclopedic knowledge.
3. Combine both sources into organized research notes.
4. Focus on accuracy, important insights, challenges, and future scope.
5. Do NOT write the final formatted report yet. Only produce research notes.
"""

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_prompt)]}
    )

    # Extract final agent text
    messages = result.get("messages", [])
    research_notes = ""
    if messages:
        research_notes = messages[-1].content
    else:
        research_notes = "No research notes generated."

    final_report = build_final_report(topic, research_notes)
    return final_report


# -----------------------------
# Save Report
# -----------------------------
def save_report(topic: str, report: str) -> str:
    # Clean topic for filename
    safe_topic = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in topic)
    safe_topic = safe_topic.strip().replace(" ", "_")

    # Create outputs folder if not exists
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Create filename with timestamp (better for multiple runs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{safe_topic}_report_{timestamp}.txt"

    # Save file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    return str(filename)


# -----------------------------
# Main
# -----------------------------
def main():
    check_env()

    print("=" * 60)
    print("Autonomous Research Agent (LangChain + Groq)")
    print("=" * 60)

    topic = input("Enter your research topic: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    print("\nResearching... Please wait.\n")
    report = run_research_agent(topic)

    # Optional: replace placeholders in cover page style
    today = datetime.now().strftime("%d-%m-%Y")
    report = report.replace("Student Name", "Ayushi Tiwari")
    report = report.replace("Submission Date", today)
    report = report.replace("Course Name", "Agentic AI / LangChain Assignment")
    report = report.replace("Assignment Name", "Assignment 2: Autonomous Research Agent")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)

    filename = save_report(topic, report)
    print(f"\nReport saved as: {filename}")


if __name__ == "__main__":
    main()