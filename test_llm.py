from llm import LLM

llm = LLM()

print("AI Assistant Ready!")

while True:
    question = input("\nYou: ")

    if question.lower() == "quit":
        break

    answer = llm.generate(question)

    print("\nAssistant:")
    print(answer)