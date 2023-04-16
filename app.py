import pickle
import streamlit as st

infile = open('ClassPick', 'rb')
lr = pickle.load(infile)
infile.close()

st.title('Language Classification')
st.write('This app can classify upto 17 Languages  (English, French, Spanish, Hindi, Portugeese, Italian, Russian, '
             'Swedish, Malayalam, Dutch, Arabic, Turkish, German, Tamil, Danish, Kannada, Greek)')

title = st.text_input("Enter the Language to Classify")
if st.button('Classify'):
     st.success('The Language is '+lr.predict([title])[0])