import os 
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
bot=Groq(api_key=os.getenv("GROQ_API_KEY"))
while True:

    inp=input("How can i help you today , sir?")
    if inp.lower() in ["exit","bye","sleep","shutdown"]:
        print("Pleaure working for you")
        break
    answer=bot.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"You are an intelligent and helpful assistant"},
            {"role":"user","content":inp}
        ],
        temperature=0.4,
        max_tokens=300
    )
    print("Vision:",answer.choices[0].message.content)
