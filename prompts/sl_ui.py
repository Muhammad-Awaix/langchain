from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv 
import os 
load_dotenv()
import streamlit as st

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model = ChatGroq(api_key=GROQ_API_KEY, 
                 model='llama-3.1-8b-instant',
                 temperature=0.7)

st.header("Research Assistant")

paper_input = st.selectbox("Select a paper to analyze", ["Attentation is all you need ", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis", "AlphaFold: Highly accurate protein structure prediction with AlphaFold"])

style_input = st.selectbox("Select a style for the output", ["Bullet Points", "Summary", "Key Takeaways", "Detailed Explanation", "Beginner Friendly", "Code Examples", "Mathematical Explanation"])

input_length = st.selectbox("Select the desired length of the output", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (6+ paragraphs)"])

template = load_prompt("paper_summary_template.json")

# using the chain concept to connect the prompt template and the model
if st.button("Submit"):
    chain = template | model
    result = chain.invoke(
        {
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': input_length

        }
    )
    st.write(result.content)