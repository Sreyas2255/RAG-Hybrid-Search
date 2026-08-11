import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET GROQ API KEY
# ============================================================

api_key = os.getenv(
    "GROQ_API_KEY"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not api_key:

    raise ValueError(
        "GROQ_API_KEY is not set in .env"
    )


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    prompt: str,
) -> str:

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# GROQ TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )

    answer = generate_answer(
        question
    )

    print(
        "\n=============================="
    )

    print(
        "GROQ RESPONSE"
    )

    print(
        "=============================="
    )

    print(
        answer
    )

