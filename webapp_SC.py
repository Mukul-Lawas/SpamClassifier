import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')

ps = PorterStemmer()


def transform_sms(sms):
    sms = sms.lower()
    sms = nltk.word_tokenize(sms)

    lst = []
    for i in sms:
        if i.isalnum():
            lst.append(i)

    sms = lst[:]
    lst.clear()

    for i in sms:
        if i not in stopwords.words('english') and i not in string.punctuation:
            lst.append(i)

    sms = lst[:]
    lst.clear()

    for i in sms:
        lst.append(ps.stem(i))

    return " ".join(lst)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_sms(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    vector_input = vector_input.toarray()
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

