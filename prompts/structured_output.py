from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model = ChatGroq(api_key=GROQ_API_KEY, 
                 model='llama-3.3-70b-versatile',  # ✅ stronger model
                 temperature=0.5)

class Review(TypedDict):
    key_themes: Annotated[list[str],"List all the key themes mentioned in the review"]
    summary: Annotated[str,"A concise summary of the review"]
    sentiment: Annotated[str,"The sentiment of the review (positive, negative, or neutral)"]
    pros: Annotated[Optional[list[str]],"List the pros mentioned in the review, if any"]
    cons: Annotated[Optional[list[str]],"List the cons mentioned in the review, if any"]

srt_model = model.with_structured_output(Review)

result = srt_model.invoke("""
    I recently upgraded to the latest version of the software, and I must say, it's fantastic! The new features are incredibly useful, and the performance has improved significantly. I had a minor issue during installation, but the customer support was quick to help me resolve it. Overall, I'm very satisfied with this update and would highly recommend it to others.
    However, I did encounter a small bug when trying to export my data, but I'm confident that the developers will address it in the next patch. Despite this minor hiccup, my overall experience has been positive, and I'm excited to see how the software continues to evolve in the future.
""")

print(result)
# print(result['summary'])
# print(result['sentiment'])