from functions import *
from predict import *
from math import log10

if 'formVisible' not in st.session_state:
    st.session_state['formVisible'] = True

if 'resultVisible' not in st.session_state:
    st.session_state['resultVisible'] = False

if st.session_state['formVisible']:
    st.session_state['age'] = st.slider("Age:", 15, 120, 15)
    st.session_state['job'] = st.selectbox("Job:", options=["admin.", "blue-collar", "entrepreneur", "housemaid",
                                                            "management", "retired", "self-employed", "services",
                                                            "student", "technician", "unemployed"])
    st.session_state['marital'] = st.selectbox("Marital:", options=["divorced", "married", "single", "widowed"])
    st.session_state['education'] = st.selectbox("Education:",
                                                 options=["illiterate", "basic.4y", "basic.6y", "basic.9y",
                                                          "high.school", "professional.course", "university.degree"])
    st.session_state['month'] = st.slider("Month:", 1, 12, 1)
    st.session_state['day_of_the_week'] = st.slider("Day of the week:", 1, 5, 1)
    st.session_state['default'] = st.radio("Has credit in default?", options=["no", "yes", "unknown"], horizontal=True)
    st.session_state['housing_loan'] = st.radio("Has housing loan", options=["no", "yes", "unknown"], horizontal=True)
    st.session_state['personal_loan'] = st.radio("Has personal loan?", options=["no", "yes", "unknown"],
                                                 horizontal=True)
    st.session_state['contact_type'] = st.radio("Contact communication type", options=["cellular", "telephone"],
                                                horizontal=True)
    st.session_state['contacted_before'] = st.radio("Client contacted before?", options=["no", "yes"],
                                                    horizontal=True)

    if st.session_state['contacted_before'] == "yes":
        st.session_state['last_contact'] = st.number_input("Number of days that passed by after the client was last "
                                                           "contacted", step=1)
        st.session_state['contacts_in_current_campaign'] = st.number_input("Number of contacts performed (in this "
                                                                           "campaign)", step=1)
        st.session_state['contacts_in_previous_campaigns'] = st.number_input("Number of contacts performed (before "
                                                                             "this campaign)", step=1)
        st.session_state['outcome'] = st.radio("Outcome of the previous marketing campaign",
                                               options=["failure", "nonexistent", "success"], horizontal=True)
    else:
        st.session_state['last_contact'] = 999
        st.session_state['contacts_in_current_campaign'] = 0
        st.session_state['contacts_in_previous_campaigns'] = 0
        st.session_state['outcome'] = "nonexistent"

    sendButton = st.button("Send!", on_click=get_predict)

if st.session_state['resultVisible']:
    query = pd.DataFrame({
        'age': [st.session_state['age']],
        'job': [st.session_state['job']],
        'marital': ["divorced" if st.session_state['marital'] == "widowed" else st.session_state['marital']],
        'education': [replace_education(st.session_state['education'])],
        'default': [st.session_state['default']],
        'housing': [st.session_state['housing_loan']],
        'loan': [st.session_state['personal_loan']],
        'contact': [st.session_state['contact_type']],
        'month': [st.session_state['month']],
        'day_of_week': [st.session_state['day_of_the_week']],
        'campaign': [st.session_state['contacts_in_current_campaign']],
        'pdays': [log10(st.session_state['last_contact'])],
        'previous': [st.session_state['contacts_in_previous_campaigns']],
        'poutcome': [st.session_state['outcome']],
        'not_contacted_before': [False if st.session_state['contacted_before'] == "yes" else True]
        })

    col1, col2, col3 = st.columns(3)
    with col1:
        column_predictions("SVC", "pip1_model.pkl", query)
    with col2:
        column_predictions('DecisionTreeClassifier', 'pip2_model.pkl', query)
    with col3:
        column_predictions('GaussianNB', 'pip3_model.pkl', query)

    final_prediction('main_model.pkl', query)

    st.toast("SVM may choose other models as an answer than those with a probability greater than 50% "
             "https://scikit-learn.org/stable/modules/svm.html#scores-probabilities")
