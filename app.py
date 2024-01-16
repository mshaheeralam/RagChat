# package the semantic search into an API with front end
# set up a flask app below

import streamlit as st
import utils

global FILE_NAME
FILE_NAME = 'tgg.parquet'
EMBEDDINGS_MODEL = "text-embedding-ada-002"
CHAT_COMPLETIONS_MODEL = "gpt-3.5-turbo-0301"

def main():

    with st.expander('State'):
        st.write(st.session_state)

    with st.form(key='question_form'):
        st.text_input('Enter your question', key='question')
        submitted = st.form_submit_button(label='Submit')

        if submitted and st.session_state.get('question'):
            prompt_obj = utils.construct_completions_prompt_exp(st.session_state.question) 
            answer = utils.get_answer_exp(prompt_obj)
            st.write(answer)
    
if __name__ == "__main__":
    main()