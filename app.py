import glob
import os
import streamlit as st
import traceback
import utils

global FILE_NAME
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
        with st.spinner('Converting to JSON...'):
            utils.pdf_to_json_format(st.session_state.file.name)
        with st.spinner('Generating embeddings...'):
            utils.generate_embeddings()
            utils.create_parquet_file()
        st.success('Embeddings generated successfully!')
        
    except Exception as e:
        st.error(e)
        print(traceback.format_exc())

def main():

    with st.expander('State'):
        st.write(st.session_state)

    if 'file_path' not in st.session_state:
        with st.container():
            st.file_uploader('Upload a PDF', type='pdf', key='file', on_change=upload)

    if st.session_state.get('file_path'):
        with st.form(key='question_form'):
            st.text_input('Enter your question', key='question')
            submitted = st.form_submit_button(label='Submit')

            if submitted and st.session_state.get('question'):
                prompt_obj = utils.construct_completions_prompt_exp(st.session_state.question) 
                answer = utils.get_answer_exp(prompt_obj)
                st.write(answer)
        
if __name__ == "__main__":
    main()