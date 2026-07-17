import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from rich.progress import track
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key not set")

client = Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"

# Pydantic Schema for customer support ticket
class Ticket(BaseModel):
    category: str
    priority: str
    sentiment: str
    summary: str
    suggested_reply: str

ticket_schema = Ticket.model_json_schema()

system_prompt=f"""
You are an expert and professional customer support manager at an e-commerce company.
You have to analyze the customer message and classify it.

You MUST return a JSON object with EXACTLY these keys and value types:
- category (string): one of Billing, Tech, Delivery, Refund, Other
- priority (string): one of Low, Medium, High
- sentiment (string): one of Angry, Neutral, Happy
- summary (string): max 2 lines, no \\n or special characters, continuous text
- suggested_reply (string): 1 line only

Return ONLY the JSON object with these fields filled in based on the customer's message.
Do NOT return the schema definition. Do NOT include a "properties" key.

Example output:
{{"category": "Billing", "priority": "High", "sentiment": "Angry", "summary": "Customer was double-charged for order. They want an immediate refund.", "suggested_reply": "We're sorry for the double charge — we'll refund the extra amount within 3-5 business days."}}
"""

response_format={
    "type": "json_object"
}



def process_ticket(content):

    system_message={
        "role": "system",
        "content": system_prompt
    }

    user_message={
        "role": "user",
        "content": content
    }

    messages = [system_message, user_message]
    response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)

    answer = response.choices[0].message.content
    # print(answer)

    return Ticket.model_validate_json(answer)

def calculate_score(ticket):
    priority_map={
        "High": 3,
        "Medium": 2,
        "Low": 1
    }
    sentiment_map = {
        "Angry": 3, 
        "Neutral": 2, 
        "Happy": 1
    }
    return priority_map[ticket.priority] + sentiment_map[ticket.sentiment]

# Pointer to tickets folder
folder = Path("tickets")
results = []

for file in track(folder.glob("*.txt"), description="Processing Tickets..."):
    content = file.read_text()

    ticket = process_ticket(content)
    score = calculate_score(ticket)

    results.append((score, file.name, ticket))



results.sort(key=lambda x: x[0], reverse=True)

table = Table(
    title="Customer Support Ticket Analysis",
    box=box.ROUNDED,
    show_lines=True
)

table.add_column("File", style="cyan", no_wrap=True)
table.add_column("Category", style="green")
table.add_column("Priority", style="yellow")
table.add_column("Sentiment", style="red")
table.add_column("Score", justify="center")
table.add_column("Summary")

for score, name, ticket in results:
    table.add_row(
        name,
        ticket.category,
        ticket.priority,
        ticket.sentiment,
        str(score),
        ticket.summary,
    )

console.print(table)