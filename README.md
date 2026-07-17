# Customer Ticket Classifier

Classifies customer support tickets using an LLM and ranks them by urgency.

## What it does

- Reads `.txt` tickets from the `tickets/` folder.
- Uses Groq (LLaMA 3.3 70B) to classify each ticket into `category`, `priority`, `sentiment`, `summary`, and `suggested_reply`.
- Scores each ticket from priority + sentiment and displays a sorted table using `rich`.

## Setup

```bash
uv sync
cp .env.example .env
# fill in your GROQ_API_KEY in .env
```

## Run

```bash
python ticket.py
```

## Stack

- Python 3.13
- Groq API
- Pydantic
- Rich
