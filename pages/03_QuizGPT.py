import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
import openai

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None

if "difficulty" not in st.session_state:
    st.session_state["difficulty"] = "easy"

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini-2024-07-18",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=st.session_state["OPENAI_API_KEY"],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)


st.title("QuizGPT")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
    
    Based ONLY on the following context, make 10 (TEN) {difficulty} questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Context: {context}
""",
        )
    ]
)

questions_chain = questions_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    input_data = {"context": format_docs(_docs), "difficulty": difficulty}
    response = questions_chain.invoke(input_data)
    response = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(response)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

def check_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError:
        return False
    else:
        return True


with st.sidebar:
    docs = None
    topic = None
    st.link_button("Github Repo", "https://github.com/DiZZi-bot/fullstack-gpt")
    if not st.session_state["OPENAI_API_KEY"]:
        OPENAI_API_KEY = st.text_input("Input your OpenAI api key", type="password")
        if OPENAI_API_KEY and check_api_key(OPENAI_API_KEY):
            st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY
            st.success("Valid API KEY.\n\nYou can upload your file.")
        elif OPENAI_API_KEY:
            st.warning("Invalid API KEY.\n\nPlease Check your API KEY")
    else:
        st.success("Valid API KEY.\n\nYou can upload your file.")

    if st.session_state["OPENAI_API_KEY"]:
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        st.session_state["difficulty"] = st.selectbox(
            "Select difficulty level:",
            ["easy", "medium", "hard"]
        )

        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini-2024-07-18",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=st.session_state["OPENAI_API_KEY"],
        ).bind(
            function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
        )
    response = run_quiz_chain(docs, topic if topic else file.name, st.session_state["difficulty"])

    total_questions = len(response["questions"])
    correct_answers = 0

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                correct_answers+=1
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()


    if button:
        if correct_answers == total_questions:
            st.success(f"Perfect! You got all {correct_answers}/{total_questions} questions correct!")
            st.balloons()
        else:
            st.error(f"You got {correct_answers}/{total_questions} correct. Try again!")
    
    if button and correct_answers != total_questions:
        retry_button = st.button("Retry Quiz")
        if retry_button:
            st.experimental_rerun() 