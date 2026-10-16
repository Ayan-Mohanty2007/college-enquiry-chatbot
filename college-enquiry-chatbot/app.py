import os

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="College Enquiry Chatbot",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# 2. APPLICATION HEADER
# --------------------------------------------------

st.title("🎓 College Enquiry Chatbot")

st.write(
    "Welcome to BIT Raipur College Enquiry System!"
)

st.caption(
    "Ask questions about courses, admission, fees, "
    "hostel, scholarships and placements."
)


# --------------------------------------------------
# 3. LOAD AND VALIDATE FAQ DATASET
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_faqs():
    file_path = os.path.join(BASE_DIR, "faqs.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "faqs.csv was not found in the project folder."
        )

    df = pd.read_csv(file_path).fillna("")

    required_columns = {
        "question",
        "answer",
        "category"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain question, answer and category columns."
        )

    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # Remove rows without questions or answers
    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ].reset_index(drop=True)

    return df


try:
    faqs = load_faqs()

    if faqs.empty:
        st.error("No valid FAQs found in faqs.csv.")
        st.stop()

    # --------------------------------------------------
    # 4. TF-IDF VECTORIZATION
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    faq_matrix = vectorizer.fit_transform(
        faqs["question"]
    )

except (FileNotFoundError, ValueError) as error:
    st.error(f"FAQ dataset error: {error}")
    st.stop()


# --------------------------------------------------
# 5. CHAT HISTORY
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# Display previous messages in the current session
for message in st.session_state.history:

    with st.chat_message("user"):
        st.write(message["question"])

    with st.chat_message("assistant"):
        st.write(message["answer"])


# --------------------------------------------------
# 6. USER QUESTION INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask about courses, admission, fees or hostel..."
)


# --------------------------------------------------
# 7. COSINE SIMILARITY MATCHING
# --------------------------------------------------

if question and question.strip():

    clean_question = question.strip()

    query_vector = vectorizer.transform(
        [clean_question]
    )

    similarity_scores = cosine_similarity(
        query_vector,
        faq_matrix
    )[0]

    best_index = int(similarity_scores.argmax())

    best_score = float(
        similarity_scores[best_index]
    )

    # Initial threshold suggested in the project guide
    SIMILARITY_THRESHOLD = 0.25

    if best_score >= SIMILARITY_THRESHOLD:

        answer = str(
            faqs.loc[best_index, "answer"]
        )

        category = str(
            faqs.loc[best_index, "category"]
        )

        answer = f"**Category:** {category}\n\n{answer}"

    else:

        answer = (
            "I could not find a sufficiently similar "
            "verified answer in my FAQ dataset. "
            "Please contact the college office for "
            "reliable information."
        )

    # Save question and answer in current-session history
    st.session_state.history.append(
        {
            "question": clean_question,
            "answer": answer
        }
    )

    # Refresh to display the new messages
    st.rerun()


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Information is based on the available FAQ dataset. "
    "Please verify important details with official "
    "college notices or the college office."
)
