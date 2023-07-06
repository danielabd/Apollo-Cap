from language_tool_python import LanguageTool

# Create a LanguageTool object for the English language
tool = LanguageTool('en-US')

# The text to be checked
text = "This sentnce has a typo and a grammer error. It's very bad."
ours_text = "This rose jar is gorgeous"
cap_dec_text = "a beautiful vase filled with beautiful flowers on top of a table"

ours_text = "A selection of delicious pizza recipes."
cap_dec_text = "A delicious pizza with lots of cheese and veggies on a plate."
# The texts to be checked
texts = [ours_text,cap_dec_text]*743

# Check each text for errors using LanguageTool
texts = ["bla bla dadasd","The beautiful kitchen toolshed.","The poor housekeeper who was forced to flee the family home after being told"]
scores = []
text = "Girl dsdasd tall tree"
for text in texts:
    matches = tool.check(text)
    score = 0
    for match in matches:
        if match.ruleId == 'TYPOGRAPHY':
            score += 2
        elif match.ruleId == 'GRAMMAR':
            score += 3
        else:
            score += 1
    scores.append(score)

# Print the scores for each text
for i, score in enumerate(scores):
    print("The score for text", i+1, "is:", score)