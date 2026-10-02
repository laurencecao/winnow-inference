from typesafe_sdk import TypeSafeClient
import os
import time

MYAPI="http://localhost:8091"
MYKEY="aaa"
MODEL="Winnow-12B"
os.environ['TYPESAFE_BASE_URL']=MYAPI

questions = {
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        "team":   {"type": "choice", "instructions": "Which team should handle it?",
                   "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
        "tone":   {"type": "score", "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "furious"]},
}
state = "Everything is down and we have a demo with our biggest client at noon.",

client = TypeSafeClient(
           api_key=MYKEY,
           base_url=MYAPI,
         )

ts = time.time()
r = client.system_one(
    model = MODEL,
    state = state,
    questions = questions,
)
ts = time.time() - ts
    
print(f"total time elapsed: {ts}s")
print(f"state is: {state}")

print(questions["urgent"])
print("urgent.noul", r.nouls["urgent"].noul)        # 1.00
print(questions["team"])
print("team.choice", r.choices["team"].choice)      # "outage", confidence 1.00
print(questions["tone"])
print("tone.score", r.scores["tone"].score)        # 2.00 (expected level, 0-indexed)
