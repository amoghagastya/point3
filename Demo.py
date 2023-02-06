import streamlit as st
import pinecone
import openai
# from openai.embeddings_utils import get_embedding
import json
import os
# from streamlit_chat import message as st_message
import streamlit.components.v1 as components  # Import Streamlit
from streamlit_chat import message


st.set_page_config(
    page_title="Hello",
    page_icon="🤖",
    layout = "centered"
)
st.markdown("""
<link
  rel="stylesheet"
  href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"
/>
""", unsafe_allow_html=True)
st.subheader("Chat with your Molton Brown Assistant!")


if 'bot' not in st.session_state:
    st.session_state['bot'] = ["Greetings! I'm your Molton Brown assistant. How may I help you today?"]

if 'user' not in st.session_state:
    st.session_state['user'] = ["Hi"]

if 'convo' not in st.session_state:
    st.session_state['convo'] = ["AI: Greetings! I'm your Molton Brown assistant. How may I help you today?"]

INDEX = 'point3-semantic-search'

def load_index():
    pinecone.init(
        api_key=st.secrets["PINECONE_KEY"],  # app.pinecone.io
        environment='us-west1-gcp'
    )
    index_name = 'point3'
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")

    return pinecone.Index(index_name)

# index = load_index()

@st.experimental_singleton(show_spinner=False)
def init_key_value():
    with open('point3-mapping.json', 'r') as fp:
        mappings = json.load(fp) 
    return mappings

with open('point3-mapping.json', 'r') as fp:
        mappings = json.load(fp) 

openai.api_key = st.secrets["OPENAI_KEY"]
    
def get_embedding(text, engine="curie-search-query"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=engine)['data'][0]['embedding']

def create_context(question, index, mappings, max_len=3750, size="curie"):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine='curie-search-query')
    res = index.query(q_embed, top_k=3, include_metadata=True)
    
    cur_len = 0
    contexts = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "\n\n###\n\n".join(contexts)

instructions = {
    "conservative Q&A": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    "paragraph about a question":"Write a paragraph, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nParagraph long Answer:",
    "bullet point": "Write a bullet point list of possible answers, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nBullet point Answer:",
    "summarize problems given a topic": "Write a summary of the problems addressed by the questions below\"\n\n{0}\n\n---\n\n",
    "extract key libraries and tools": "Write a list of libraries and tools present in the context below\"\n\nContext:\n{0}\n\n---\n\n",
    "just instruction": "{1} given the common questions and answers below \n\n{0}\n\n---\n\n",
    "summarize": "Write an elaborate, paragraph long summary about \"{1}\" given the questions and answers from a public forum on this topic\n\n{0}\n\n---\n\nSummary:",
    "chat": "The following is a chat conversation between an AI Molton Brown customer support assistant and a user. Write a paragraph, addressing the question, and use the text below to obtain relevant information. If question absolutely cannot be answered based on the context, say \"I don't know, sorry\". Allow for chit chat with the user and end the conversation when user is no longer interested.\n\nContext:\n{0}\n\n---\n\nChat: {1}"
}

convo = st.session_state['convo']

def chat(
    index,
    fine_tuned_qa_model="text-davinci-003",
    question="i need a sun screen product",
    instruction="Answer the query based on the context below, and if the query can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    max_len=3550,
    size="ada",
    debug=False,
    max_tokens=400,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    convo.append("User: " + question)
    # remove convo newlines and concat to ctx
    ctx =  ('').join(("\n").join(convo).splitlines())
    context = create_context(
        ctx,
        index,
        mappings,
        max_len=max_len,
        size=size,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        print('convo so far ', convo)
        print(instruction.format(context, question))
        response = openai.Completion.create(
            prompt=instruction.format(context, ("\n").join(convo)),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            **model_param,
        )
        convo.append(response["choices"][0]["text"].strip())
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

def generate_ans(user_input):
    print('generating answer...')
    st.session_state.user.append(user_input)
    st.session_state.convo.append("User: " + user_input)
    ctx = ('\n').join(st.session_state.convo)
    print('convo so far', st.session_state.convo)
    st.session_state.convo.append("AI: ")
    result = chat(index, question=user_input, 
                            instruction = instructions["chat"], debug=False) 
    return st.session_state.bot.append(result)

# user_input = get_text()
with st.spinner("Connecting to OpenAI..."):
    openai.api_key = st.secrets["OPENAI_KEY"]

with st.spinner("Connecting to Pinecone..."):
    index = load_index()
    text_map = init_key_value()

def clear_text():
    st.session_state["text"] = ""
        
def main():
    search = st.container()
    query = search.text_input('Ask a product related question!', value="", key="text")
    
    # with search.expander("Chat Options"):
    #     style = st.radio(label='Style', options=[
    #         'Conservative Q&A',
    #         'Chit-chat Allowed'
    #     ],on_change=clear_text)  
    
    # search.button("Go!", key = 'go')
    if search.button("Go!") or query != "":
        with st.spinner("Retrieving, please wait..."):
            # lowercase relevant lib filters
            # ask the question
            answer = generate_ans(query)
            # clear_text()            
            # return 
    if st.button("Reset Chat", on_click=clear_text):
        st.session_state['bot'] = ["Greetings! I'm your Molton Brown assistant. How may I help you today"]
        st.session_state['user'] = ["Hi"]
        st.session_state['convo'] = ["AI: Greetings! I'm your Molton Brown assistant. How may I help you today"]
        
    if st.session_state['bot']:
            for i in range(len(st.session_state['bot'])-1, -1, -1):
                message(st.session_state["bot"][i], key=str(i))
                message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')

main()
