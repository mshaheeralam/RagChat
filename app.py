import glob
import os
import streamlit as st
import traceback
import utils

FILE_NAME = 'tgg.parquet'
EMBEDDINGS_MODEL = "text-embedding-ada-002"
CHAT_COMPLETIONS_MODEL = "gpt-3.5-turbo-0301"

def upload():
    try:
        files = glob.glob("*.pdf")
        for f in files:
            os.remove(f)
        with open(st.session_state.file.name, "wb") as f:
            f.write(st.session_state.file.getbuffer())
        st.session_state["file_path"] = st.session_state.file.name
        with st.spinner('Generating embeddings...'):
            utils.pdf_to_json_format(st.session_state.file.name)
            utils.generate_embeddings(st.session_state.api_key)
            utils.create_parquet_file()
        st.success('Embeddings generated successfully!')
        
    except Exception as e:
        st.error(e)
        print(traceback.format_exc())

def main():
    # with st.expander("State"):
    #     st.session_state

    try:
        if 'api_key' not in st.session_state:
            with st.form(key='api_key_form'):
                api_key = st.text_input("Enter your OpenAI API key") 
                submitted = st.form_submit_button(label='Submit')
                if submitted:
                    st.session_state['api_key'] = api_key
                    st.success('API key saved!')

        if 'file_path' not in st.session_state and st.session_state.get('api_key'):
            with st.container():
                st.file_uploader('Upload a PDF', type='pdf', key='file', on_change=upload)

        if st.session_state.get('file_path'):
            st.chat_input('Enter your question', key='question')
            if st.session_state.get('question'):
                prompt_obj = utils.construct_completions_prompt_exp(st.session_state.question, st.session_state.api_key) 
                answer = utils.get_answer_exp(prompt_obj, st.session_state.api_key)
                st.write(answer)
    except Exception as e:
        st.error(e)
        print(traceback.format_exc())

if __name__ == "__main__":
    main()