import streamlit as st
import cv2
# import streamlit_authenticator as stauth
import numpy as np
import pandas as pd
import tempfile
import webbrowser
from Fightdet.predict import make_prediction

st.set_page_config(page_title="VDS (v1.0)",
    page_icon="👊",
    layout="wide",  # wide
    initial_sidebar_state="auto")

col1, col2, col3 = st.columns([7,1,1.2])



if col3.button("👨 Login/Sign Up"):
    # print is visible in the server output, not in the page
    names = st.text_input('Name', 'Chew')
    usernames = st.text_input('Username', 'Fightclub')
    password = st.text_input('Password','alphanumerical only')
    st.write('I was clicked 🎉')
if col2.button("🏠 Home Page"):
    webbrowser.open_new_tab("www.google.com")

col1.title("""Violence Detection System""")

uploaded_file = st.file_uploader("Please upload your video file", type=["mp4","avi"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1,1,1])
    col2.video(uploaded_file, start_time=0)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    progress = st.markdown(f"making prediction.......")
    result=make_prediction(tfile.name)
    if result==0:
        st.error("Violence Activity Detected")
    else:
        st.success("No Violence Activity Detected")

    progress.empty()

    #fps display
    # vf = cv2.VideoCapture(tfile.name)
    # st.markdown(f"video fps : {int(vf.get(cv2.CAP_PROP_FPS))}")

st.info('''Higher quality video clip length will will require much higher time for prediction.''')


# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider

agree = st.checkbox('Program Customization')

if agree:
    line_count = st.slider('Selection Threshold', 1, 10, 3)

# and used in order to select the displayed lines
st.write("")
st.write(""":book: Background  \n"""
    """This study is inspired by the works of Cheng, Cai, and Li's work in RWF-2000: An Open Large Scale Video Database \
    for Violence Detection in 2019.  \n Further exploration based on their study is conducted by testing on several different \
    models, with different channels and optimizers.  \n For this demonstration, the model is developed based on grayscale \
    + optical flows, utilizing the Flowed Gated Network architecture.""")

st.write("")
st.write(
    ''':information_source: Disclaimer  \n'''
    '''This frontend is developed for showcasing model's capability, therefore this presentation are done by \
    manual uploading video as shown instead of direct connection through streaming input''')


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by
<span><a style='display: text-align: center;' href="https://github.com/Chewgithub" target="_blank"> EY Chew </a> | \
<a style='display: text-align: center;' href="https://github.com/yc-ng" target="_blank"> YC Ng </a> | \
<a style='display: text-align: center;' href="https://github.com/yongsin91/" target="_blank"> YS Tan </a></span>
<a text-align: right>
VDS v1.0 - Last Update: 14-6-2022</a></p></div>
"""
st.markdown(footer,unsafe_allow_html=True)

# # Login System
# names = ['John Smith','Rebecca Briggs']
# usernames = ['jsmith','rbriggs']
# passwords = ['123','456']

# hashed_passwords = stauth.Hasher(passwords).generate()
# authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
#     'some_cookie_name','some_signature_key',cookie_expiry_days=30)
# if st.session_state['authentication_status']:
#     authenticator.logout('Logout', 'main')
#     st.write('Welcome *%s*' % (st.session_state['name']))
#     st.title('Some content')
# elif st.session_state['authentication_status'] == False:
#     st.error('Username/password is incorrect')
# elif st.session_state['authentication_status'] == None:
#     st.warning('Please enter your username and password')
