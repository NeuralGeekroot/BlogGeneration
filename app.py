import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, TypedDict, Annotated
from langgraph.constants import Send
import operator
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from openai import OpenAI

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGSMITH_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT_NAME'] = os.getenv('LANGCHAIN_PROJECT_NAME')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize LLM
llm = ChatGroq(model='gemma2-9b-it') 

# Initialize OpenAI client for DALL·E
client = OpenAI()

# Define section structure
class Section(BaseModel):
    section_name: str = Field(description="Section name")
    description: str = Field(description="Description of the section")

class Sections(BaseModel):
    sections: List[Section] = Field(description="List of section details")

structured_sections = llm.with_structured_output(Sections)

# Define blog state
class BlogState(TypedDict):
    topic: str
    outline: str
    sections: list[Section]
    completed_section: Annotated[list, operator.add]
    review_content: str
    send_seo_optimization: str
    revise_section_content: list[str]
    finalize_blog: str
    step: str
    final_blog: str
    image_urls: list  
    fallback_links: list 

class BlogStateSection(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

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

# Orchestrator node to generate an outline
@traceable
def generate_outline(state: BlogState):
    st.write("Generating an outline for the blog...")
    result = structured_sections.invoke([
        SystemMessage(content="Provide an interesting and informative content outline for the given {topic}."),
        HumanMessage(content=f"Here is the blog topic: {state['topic']}")
    ])
    return {'topic': state['topic'], 'outline': result.sections}

# Worker node to write sections
@traceable
def write_section(state: BlogStateSection):
    st.write("Generating content for the section...")
    section_content = llm.invoke([
        SystemMessage(content="Write a detailed blog section based on the provided name and description."),
        HumanMessage(content=f"Section Name: {state['section'].section_name}, Description: {state['section'].description}")
    ])
    return {"completed_section": [section_content.content]}

# Function to generate an image using DALL·E
@traceable
def generate_image(state: BlogState):
    st.write("Generating an image for the section...")
    if not state.get('completed_section'):
        st.warning("No completed sections found to generate an image.")
        return {"image_urls": state.get('image_urls', []), "fallback_links": state.get('fallback_links', [])}
    
    section = state['topic']
    prompt = f"Generate an image for the blog section: {section} with no text. More of a representation and informative image"
    
    image_url = generate_image_with_dalle(prompt)
    if image_url:
        image_urls = state.get('image_urls', [])
        image_urls.append(image_url)
        return {"image_urls": image_urls, "fallback_links": state.get('fallback_links', [])}
    else:
        fallback_links = state.get('fallback_links', [])
        fallback_link = get_fallback_image_link(state['topic'])
        fallback_links.append(fallback_link)
        return {"image_urls": state.get('image_urls', []), "fallback_links": fallback_links}

# Review node to check the quality of sections
@traceable
def review_section(state: BlogState):
    st.write("Reviewing the section...")
    prompt = PromptTemplate.from_template(
        "Check if the section can be improved: {completed_section}. "
        "If no, return 'send_seo_optimization'. "
        "If yes, return 'revise_section_content'."
    )
    chain = prompt | llm
    result = chain.invoke({'completed_section': state['completed_section']})

    decision = result.content.strip().lower()
    if decision not in ["send_seo_optimization", "revise_section_content"]:
        decision = "send_seo_optimization"

    return {"step": decision}

# Revision node to improve content
@traceable
def revise_section(state: BlogState):
    st.write("Revising the section content...")
    if state['step'] == "revise_section_content":
        if not state.get('sections'):
            st.warning("No sections found to revise.")
            return {"completed_section": state['completed_section']}
        
        revised_content = llm.invoke([
            SystemMessage(content="Based on the review feedback, improve the content further."),
            HumanMessage(content=f"Section Name: {state['sections'][0].section_name}, Description: {state['sections'][0].description}")
        ])
        return {"completed_section": [revised_content.content]}

# Assign writers dynamically to sections
@traceable
def assign_writers(state: BlogState):
    st.write("Assigning writers to sections...")
    if not state.get('outline'):
        st.warning("No outline found to assign writers.")
        return []
    return [Send('write_section', {'section': s}) for s in state['outline']]

# Decision function for routing after review
def should_revise(state: BlogState):
    return state["step"]

# SEO Optimization step
@traceable
def seo_optimization(state: BlogState):
    st.write("Performing SEO optimization...")
    result = llm.invoke(f"Optimize the blog for search ranking: {state['topic']}")
    return {'finalize_blog': result.content}

# Final publishing step
@traceable
def publish_blog(state: BlogState):
    st.write("Finalizing and publishing the blog...")
    final_blog = state['finalize_blog']
    
    # Add images to the blog
    if state.get('image_urls'):
        st.subheader("AI-Generated Images")
        for image_url in state['image_urls']:
            st.image(image_url, caption="AI-Generated Image")
    
    # Add fallback links if images were not generated
    if state.get('fallback_links'):
        st.subheader("Fallback Image Search Links")
        for link in state['fallback_links']:
            st.markdown(f"[Search for related images on Google]({link})")
    
    return {"final_blog": final_blog}

# Build LangGraph workflow
builder = StateGraph(BlogState)

# Add orchestrator nodes
builder.add_node('generate_outline', generate_outline)

# Add worker and review nodes
builder.add_node('write_section', write_section)
builder.add_node('generate_image', generate_image)
builder.add_node('review_section', review_section)
builder.add_node('revise_section', revise_section)

# Add finalization nodes
builder.add_node('seo_optimization', seo_optimization)
builder.add_node('publish_blog', publish_blog)

# Define workflow edges
builder.add_edge(START, 'generate_outline')
builder.add_conditional_edges('generate_outline', assign_writers, ['write_section'])
builder.add_edge('write_section', 'generate_image')
builder.add_edge('generate_image', 'review_section')
builder.add_conditional_edges('review_section', should_revise, {'revise_section_content': 'revise_section', 'send_seo_optimization': 'seo_optimization'})
builder.add_edge('revise_section', 'review_section')  # Loop back after revision
builder.add_edge('seo_optimization', 'publish_blog')
builder.add_edge('publish_blog', END)

# Compile workflow
workflow = builder.compile()

# Streamlit app
def main():
    st.title("Blog Writing Assistant")
    
    # Input for blog topic
    topic = st.text_input("Enter the blog topic:")
    
    if st.button("Generate Blog"):
        if topic:
            # Define initial state
            initial_state = {
                'topic': topic,
                'outline': "",
                'sections': [],
                'completed_section': [],
                'review_content': "",
                'send_seo_optimization': "",
                'revise_section_content': [],
                'finalize_blog': "",
                'step': "",
                'final_blog': "",
                'image_urls': [],  # Initialize image URLs list
                'fallback_links': []  # Initialize fallback links list
            }
            
            # Invoke workflow
            try:
                result = workflow.invoke(initial_state)
                st.subheader("Final Blog Content")
                st.write(result['final_blog'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter a blog topic.")

if __name__ == "__main__":
    main()