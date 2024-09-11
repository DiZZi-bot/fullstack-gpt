import streamlit as st

body = '''
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import openai
import re


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini-2024-07-18",
        openai_api_key=st.session_state["OPENAI_API_KEY"],
    )
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini-2024-07-18",
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=st.session_state["OPENAI_API_KEY"],
    )
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=(
            [
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ]
        ),
        parsing_function=parse_page,
    )
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API_KEY"])

    cache_filename = re.sub(r'[\/:*?"<>|]', '_', url)
    cache_filename.strip()
    cache_dir = LocalFileStore(f"./.cache/{cache_filename}/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()

def check_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError:
        return False
    else:
        return True


with st.sidebar:
    st.link_button("Github Repo", "https://github.com/DiZZi-bot/fullstack-gpt")
    if not st.session_state["OPENAI_API_KEY"]:
        OPENAI_API_KEY = st.text_input("Input your OpenAI api key", type="password")
        if OPENAI_API_KEY and check_api_key(OPENAI_API_KEY):
            st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY
            st.success("Valid API KEY.\n\nYou can write down a Sitemap URL.")
        elif OPENAI_API_KEY:
            st.warning("Invalid API KEY.\n\nPlease Check your API KEY")
    else:
        st.success("Valid API KEY.\n\nYou can write down a Sitemap URL.")

    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if not st.session_state["OPENAI_API_KEY"]:
    st.markdown(
        """
        # SiteGPT
                
        Ask questions about the content of a website.
                
        Start by writing the URL of the website on the sidebar.
    """
    )
elif ".xml" not in url:
    with st.sidebar:
        st.error("Please write down a Sitemap URL.")
elif st.session_state["OPENAI_API_KEY"] and url:
    retriever = load_website(url)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    message = st.chat_input("Ask a question to the website.")
    if message:
        send_message(message, "human")
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
'''

st.code(body, language="python", line_numbers=True)