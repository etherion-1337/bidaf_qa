import streamlit as st
from allennlp import pretrained
import matplotlib.pyplot as plt
import numpy as np
from allennlp.predictors.predictor import Predictor

st.header("Query Engine")
st.subheader("by Zk Xav")

bidaf_path = "../bidaf-model-2020.02.10-charpad"
bidaf_elmo_path = "../bidaf-elmo-model-2018.11.30-charpad"

@st.cache(allow_output_mutation=True)
def load_model():

    serialization_dir = bidaf_path
    predictor = Predictor.from_path(serialization_dir)
    return predictor

predictor = load_model()

data = st.text_input("Context", "type your context")

# Create a text input to input the question.
question = st.text_input("question", "what would you like to know")

# Use the predictor to find the answer.
result = predictor.predict(question, data)


# From the result, we want "best_span", "question_tokens", and "passage_tokens"
start, end = result["best_span"]
question_tokens = result["question_tokens"]
passage_tokens = result["passage_tokens"]

# highlight test
mds = [f"**{token}**" if start <= i <= end else token
       for i, token in enumerate(passage_tokens)]

# And then we'll just concatenate them with spaces.
st.markdown(" ".join(mds))

# We'd also like to make a heatmap of the passage-question attention.
# We'll use plt.imshow() for that.

start, end = result['best_span']
attention = result["passage_question_attention"][start:end+1]

fig, ax = plt.subplots()
plt.imshow(attention, cmap="hot")

# Make sure to show every tick
ax.set_title("attention matrix")
ax.set_xticks(np.arange(len(question_tokens)))
ax.set_yticks(np.arange(len(attention)))

# Use the tokens as the labels
ax.set_xticklabels(question_tokens)
ax.set_yticklabels(result["passage_tokens"][start:end+1])
plt.xticks(rotation=40)

# And add it to our output
st.pyplot()