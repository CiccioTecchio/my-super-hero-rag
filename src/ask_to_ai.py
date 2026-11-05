from google import genai
import json
import re

def answer_to_JSON(answer: str) -> dict[str, str | list[str]]:
    fenced_json = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        answer,
        re.DOTALL,
    )

    if fenced_json:
        answer = fenced_json.group(1)
    return json.loads(answer)

def ask_to_LLM(
    best_document: str,
    question: str,
) -> dict[str, str | list[str]]:
    prompt = f"""Answer the question based only on this context:
Context:
{best_document}
Question: {question}
Create a JSON with the following fields:
name: string //name of the character
superheroName: string //superhero name
universe: string //DC o Marvel
superpower: string[] //list of the superpower, if is a human list his/her characteristics
"""
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )
    return answer_to_JSON(response.text.strip())