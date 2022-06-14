import streamlit as st
import cv2
import streamlit_authenticator as stauth
import numpy as np
import pandas as pd
import tempfile
import time

st.set_page_config(page_title="VDS (v1.0)",
    page_icon="🐍",
    layout="wide",  # wide
    initial_sidebar_state="auto")

if st.sidebar.button("Login/Sign Up"):
    # print is visible in the server output, not in the page
    names = st.text_input('Name', 'Chew')
    usernames = st.text_input('Username', 'Fightclub')
    password = st.text_input('Password','alphanumerical only')
    st.write('I was clicked 🎉')
st.sidebar.markdown("[Home Page](https:www.google.com)")

st.markdown("""# Violence Detection System
VDS v1.0 - Last Update: 14-6-2022""")

uploaded_file = st.file_uploader("Please upload your video file", type=["mp4","avi"])

if uploaded_file is not None:

    st.video(uploaded_file, start_time=0)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    vf = cv2.VideoCapture(tfile.name)

    st.markdown(f"video fps : {int(vf.get(cv2.CAP_PROP_FPS))}")

st.warning('''Warning! Maximum acceptable video clip length will be 5 seconds.''')

df = pd.DataFrame({
          'first column': list(range(1, 11)),
          'second column': np.arange(10, 101, 10)
        })

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
st.markdown("""Program Customization""")
line_count = st.slider('Selection Threshold', 1, 10, 3)

# and used in order to select the displayed lines
head_df = df.head(line_count)

head_df

# Login System
names = ['John Smith','Rebecca Briggs']
usernames = ['jsmith','rbriggs']
passwords = ['123','456']

hashed_passwords = stauth.Hasher(passwords).generate()
authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=30)
if st.session_state['authentication_status']:
    authenticator.logout('Logout', 'main')
    st.write('Welcome *%s*' % (st.session_state['name']))
    st.title('Some content')
elif st.session_state['authentication_status'] == False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] == None:
    st.warning('Please enter your username and password')
