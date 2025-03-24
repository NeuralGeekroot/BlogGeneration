import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import PromptTemplate
import streamlit as st
from typing import List, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.constants import Send
import operator
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import ArxivQueryRun, TavilySearchResults, YouTubeSearchTool
from langchain_community.utilities import ArxivAPIWrapper
from langchain import hub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from IPython.display import Image, display

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGSMITH_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT_NAME'] = os.getenv('LANGCHAIN_PROJECT_NAME')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# Initialize LLM and tools
llm = ChatGroq(model='gemma2-9b-it')
client = OpenAI()

# Manually initialize the TavilySearchResults tool
tavily_tool = TavilySearchResults(max_results=1)

# Load other tools
tools = [
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper()),
    YouTubeSearchTool(),
    tavily_tool,  # Add the manually initialized Tavily tool
]

prompt = hub.pull("hwchase17/react")

# Create an agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class Route(BaseModel):
    step: Literal["Arxiv", "Youtube", "Text"] = Field(
        None, description="The next step in the routing process"
    )
router = llm.with_structured_output(Route)

# Define the BlogState
class BlogState(TypedDict):
    search_results: List[dict]  # Ensure search_results is a list of dictionaries
    input_type: str
    input_data: str
    summary: List[str]
    outline: List[str]
    completed_sections: Annotated[List[str], operator.add]
    image_urls: List[str]
    fallback_links: List[str]
    review_content: str
    seo_optimized_content: str
    final_blog: str

# Router Node
@traceable
def router_node(state: BlogState):
    st.write('Deciding the router node...')
    input_type = router.invoke(
        [
            SystemMessage(
                content="""Route the input data to Arxiv, Youtube, or Text node based on the user's request.
                - If the input is an arXiv link (e.g., https://arxiv.org/abs/2106.15928) or arXiv ID (e.g., 2106.15928), route to 'Arxiv'.
                - If the input is a YouTube link (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ), route to 'Youtube'.
                - If the input is plain text (e.g., 'Latest advancements in AI and machine learning'), route to 'Text'.
                """
            ),
            HumanMessage(content=state["input_data"]),
        ]
    )
    st.write(f"LLM routing the input data to {input_type.step}")
    return {"input_type": input_type.step}

def route_decision(state):
    st.write('Routing to the specific node...')
    if state['input_type'] == 'Arxiv':
        return 'arxiv_tool'
    elif state['input_type'] == 'Youtube':
        return 'youtube_tool'
    else:
        return 'text_tool'

# Tool Nodes (Replaced with AgentExecutor)
@traceable
def arxiv_tool_node(state: BlogState):
    if state['input_type'] == 'Arxiv':
        st.write("Fetching data from arXiv using agent...")
        result = agent_executor.invoke({"input": state['input_data']})
        return {**state, 'search_results': [{"content": result['output'], "url": state['input_data']}]}
    return state

@traceable
def youtube_tool_node(state: BlogState):
    if state['input_type'] == 'Youtube':
        st.write("Fetching data from YouTube using agent...")
        result = agent_executor.invoke({"input": state['input_data']})
        return {**state, 'search_results': [{"content": result['output'], "url": state['input_data']}]}
    return state

@traceable
def text_tool_node(state: BlogState):
    if state['input_type'] == 'Text':
        st.write("Searching web for the data using agent...")
        result = agent_executor.invoke({"input": state['input_data']})
        return {**state, 'search_results': [{"content": result['output'], "url": "https://example.com"}]}
    return state

@traceable  # LangSmith debugging
def summarize_results(state: BlogState):
    """Summarizes the web search results."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Ensure search_results is a list of dictionaries
    search_results = state.get("search_results", [])
    if not isinstance(search_results, list):
        search_results = []

    # Convert search results into Document objects
    documents = [
        Document(page_content=result.get("content", ""), metadata={"source": result.get("url", "")})
        for result in search_results if result and isinstance(result, dict) and result.get("content")
    ]

    if not documents:
        summary = "No relevant information available."
    else:
        splits = text_splitter.split_documents(documents)
        summary = "\n".join(doc.page_content for doc in splits[:3])  # Taking first 3 chunks

    return {**state, 'summary': summary}

# Orchestrator Node
@traceable
def orchestrator_node(state: BlogState):
    st.write("Creating blog outline...")
    sys_msg = SystemMessage(content="Provide an interesting and informative content outline for the given summary.")
    human_msg = HumanMessage(content=f"Here is the blog topic: {state['summary']}")
    result = llm.invoke([sys_msg, human_msg])
    outline = result.content.split("\n") if isinstance(result.content, str) else result.content
    return {**state, 'outline': outline}

# Assign Writers Node
@traceable
def assign_writers(state: BlogState):
    st.write("Assigning writers to sections...")
    if not state.get('outline'):
        st.write("No outline found to assign writers.")
        return []
    return [Send('section_writer', {'section': s}) for s in state['outline']]

# Section Writer Node
@traceable
def section_writer_node(state: BlogState):
    st.write("Generating content for the section...")
    section_content = llm.invoke([
        SystemMessage(content="Write a detailed blog section based on the provided name and description."),
        HumanMessage(content=f"Section Name: {state['section']}, Description: {state['section']}")
    ])
    completed_sections = state.get("completed_sections", [])
    completed_sections.append(section_content.content)
    return {**state, "completed_sections": completed_sections}

# Function to generate an image using DALL·E
def generate_image_with_dalle(prompt: str):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"Failed to generate image: {e}")
        return None

# Function to provide a fallback link for image search
def get_fallback_image_link(topic: str):
    # Provide a Google Images search link for the topic
    search_query = topic.replace(" ", "+")
    return f"https://www.google.com/search?q={search_query}&tbm=isch"

# Image Generator Node
@traceable
def image_generator_node(state: BlogState):
    st.write("Generating an image for the section...")
    completed_sections = state.get("completed_sections", [])
    if not completed_sections:
        st.write("No completed sections found to generate an image.")
        return {**state, "image_urls": state.get('image_urls', []), "fallback_links": state.get('fallback_links', [])}
    
    section = completed_sections[0]
    prompt = f"Generate an image for the blog section: {section} with no text. More of a representation and informative image"
    
    # Use an open-source image generation model or fallback
    image_url = generate_image_with_dalle(prompt)  # Replace with open-source model
    if image_url:
        image_urls = state.get('image_urls', [])
        image_urls.append(image_url)
        return {**state, "image_urls": image_urls, "fallback_links": state.get('fallback_links', [])}
    else:
        fallback_links = state.get('fallback_links', [])
        fallback_link = get_fallback_image_link(section)
        fallback_links.append(fallback_link)
        return {**state, "image_urls": state.get('image_urls', []), "fallback_links": fallback_links}

# Review Node
@traceable
def review_node(state: BlogState):
    st.write("Reviewing the section...")
    completed_sections = state.get("completed_sections", [])
    if not completed_sections:
        st.write("No completed sections found to review.")
        return {"step": "send_seo_optimization"}
    
    prompt = PromptTemplate.from_template(
        "Check if the section can be improved: {completed_sections}. "
        "If no, return 'send_seo_optimization'. "
        "If yes, return 'revise_section_content'."
    )
    chain = prompt | llm
    result = chain.invoke({'completed_sections': completed_sections})

    decision = result.content.strip().lower()
    if decision not in ["send_seo_optimization", "revise_section_content"]:
        decision = "send_seo_optimization"

    return {"step": decision}

# SEO Optimization Node
@traceable
def seo_optimization_node(state: BlogState):
    st.write("Performing SEO optimization...")
    completed_sections = state.get("completed_sections", [])
    if not completed_sections:
        st.write("No completed sections found for SEO optimization.")
        return state
    
    result = llm.invoke(f"Optimize the blog for search ranking: {completed_sections}")
    return {**state, 'seo_optimized_content': result.content}

# Publish Node
@traceable
def publish_node(state: BlogState):
    st.write("Finalizing and publishing the blog...")
    final_blog = state.get('seo_optimized_content', '')
    
    # Add images to the blog
    if state.get('image_urls'):
        st.write("AI-Generated Images")
        for image_url in state['image_urls']:
            st.image(image_url, caption="AI-Generated Image")
    
    # Add fallback links if images were not generated
    if state.get('fallback_links'):
        st.write("Fallback Image Search Links")
        for link in state['fallback_links']:
            st.markdown(f"[Search for related images on Google]({link})")
    
    return {**state, "final_blog": final_blog}

# Streamlit App
def main():
    st.title("Blog Generation Workflow")
    
    # Input options
    input_data = st.text_input("Enter YouTube, Arxiv URL, or your desired Query")
    
    if st.button("Run Workflow"):
        # Initialize the state
        initial_state = {
            "search_results": [],  # Initialize as an empty list
            "input_type": "",  # Will be set by the router_node
            "input_data": input_data,
            "summary": [],
            "outline": [],
            "completed_sections": [],
            "image_urls": [],
            "fallback_links": [],
            "review_content": "",
            "seo_optimized_content": "",
            "final_blog": "",
        }
        
        # Build the workflow
        builder = StateGraph(BlogState)
        builder.add_node("router", router_node)
        builder.add_node("arxiv_tool", arxiv_tool_node)
        builder.add_node("youtube_tool", youtube_tool_node)
        builder.add_node("text_tool", text_tool_node)
        builder.add_node("orchestrator", orchestrator_node)
        builder.add_node("section_writer", section_writer_node)
        builder.add_node("image_generator", image_generator_node)
        builder.add_node("review", review_node)
        builder.add_node("seo_optimization", seo_optimization_node)
        builder.add_node("publish", publish_node)
        builder.add_node('summarize_results', summarize_results)
        
        # Define edges
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            route_decision,
            {
                "arxiv_tool": "arxiv_tool",
                "youtube_tool": "youtube_tool",
                "text_tool": "text_tool",
            },
        )
        builder.add_edge("arxiv_tool", "summarize_results")
        builder.add_edge("youtube_tool", "summarize_results")
        builder.add_edge('text_tool', 'summarize_results')
        builder.add_edge('summarize_results', 'orchestrator')
        builder.add_conditional_edges("orchestrator", assign_writers, ["section_writer"])
        builder.add_edge("section_writer", "image_generator")
        builder.add_edge("image_generator", "review")
        builder.add_conditional_edges(
            "review",
            lambda state: "seo_optimization" if state.get("step") == "send_seo_optimization" else "section_writer",
        )
        builder.add_edge("seo_optimization", "publish")
        builder.add_edge("publish", END)

        # Compile the workflow
        workflow = builder.compile()
        
        # Run the workflow
        result = workflow.invoke(initial_state)
        
        # Display the final result
        st.subheader("Final Blog Output")
        st.write(result['final_blog'])

if __name__ == "__main__":
    main()