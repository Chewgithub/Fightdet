import streamlit as st
# import streamlit_authenticator as stauth
import tempfile
import webbrowser
from Fightdet.predict import make_prediction
from tensorflow.keras.models import load_model
from io import BytesIO

@st.cache(allow_output_mutation=True)
def loading():
    return load_model('final_model_gray_op')

## Page Configuration
st.set_page_config(page_title="VDS (v1.0)",
    page_icon="👊",
    layout="wide",  # wide
    initial_sidebar_state="auto")

## Top Bar Configuration
col1, col2, col3 = st.columns([7,1,1.2])

if col3.button("👨 Login/Sign Up"):
    # print is visible in the server output, not in the page
    names = st.text_input('Name', 'Chew')
    usernames = st.text_input('Username', 'Fightclub')
    password = st.text_input('Password','alphanumerical only')
if col2.button("🏠 Home Page"):
    webbrowser.open_new_tab("https://github.com/Chewgithub/Fightdet")

col1.title("""Violence Detection System""")

# example videos that are pre-loaded for the demo
preload_videos = {
    'Example 1': 'demo_files/demo_01.mp4',
    'Example 2': 'demo_files/demo_02.mp4',
    'Example 3': 'demo_files/demo_03.mp4',
    'Example 4': 'demo_files/demo_04.mp4',
    'Example 5': 'demo_files/demo_05.mp4'
}

video_choice = st.radio("Select a video or Upload your own",
     ['Upload Video', 'Example 1', 'Example 2', 'Example 3', 'Example 4', 'Example 5'],
     horizontal=True)
if video_choice == 'Upload Video':
    uploaded_file = st.file_uploader("Please upload your video file", type=["mp4","avi"])
else:
    # open pre-loaded video for subsequent prediction
    with open(preload_videos[video_choice], 'rb') as con:
        uploaded_file = BytesIO(con.read())

# play the video while the model carries out the prediction
if uploaded_file is not None:
    col1, col2 = st.columns([2,1])
    col1.video(uploaded_file, start_time=0)
    with col2:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        with st.spinner(text="Making prediction......."):
            result=make_prediction(loading(), tfile.name)
    if round(result)==1:
        col2.error("**Violent Activity Detected**")
    else:
        col2.success("**No Violent Activity Detected**")
    col2.info(f"Predicted probability that violent activity is occurring: {100*result:0.2f}%")

st.info('''Higher quality and longer video clips will require longer time for prediction.''')

# Bottom info
st.markdown("""### :book: Background

This study is inspired by the works of Cheng, Cai, and Li's work in [RWF-2000: An Open Large Scale Video Database](https://arxiv.org/abs/1911.05913v3)
for Violence Detection in 2019. Further exploration based on their study is conducted by testing on several different models, with different channels
and optimizers. For this demonstration, the model is developed based on grayscale + optical flows, utilizing the Flowed Gated Network architecture.
""")

# Footer
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
<p>ℹ️ Disclaimer : This frontend is developed for showcasing model's capability, therefore the demonstration is done by
manual uploading video as shown instead of direct connection through streaming input.
<p>Developed by
<span><a style='display: text-align: center;' href="https://github.com/Chewgithub" target="_blank"> EY Chew </a> | \
<a style='display: text-align: center;' href="https://github.com/yc-ng" target="_blank"> YC Ng </a> | \
<a style='display: text-align: center;' href="https://github.com/yongsin91/" target="_blank"> YS Tan </a></span>
<span><a text-align: center>&nbsp;&nbsp;VDS v1.0 - Last Update: 14-6-2022</a></span></p></div>
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
