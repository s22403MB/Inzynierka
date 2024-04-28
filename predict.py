import streamlit as st
import joblib


def column_predictions(name, file, query):
    st.write(name)
    model = joblib.load(file)
    result = model.predict(query)[0]

    if result == 0:
        st.markdown(f"Predicted result of advertisement: <span style='color: red;'>Failure</span>",
                    unsafe_allow_html=True)
        if name == 'SVC':
            probability = model.predict_proba(query)[0] * 100
            st.markdown(f"Predicted Probability: {round(probability[0], 2)}%</div>", unsafe_allow_html=True)
            return
    else:
        st.markdown(f"Predicted result of advertisement: <span style='color: green;'>Success</span>",
                    unsafe_allow_html=True)
        if name == "SVC":
            probability = model.predict_proba(query)[0] * 100
            st.markdown(f"Predicted Probability: {round(probability[1], 2)}%</div>", unsafe_allow_html=True)
            return

    probability = round(model.predict_proba(query).max() * 100, 2)
    st.markdown(f"Predicted Probability: {probability}%</div>", unsafe_allow_html=True)


def final_prediction(file, query):
    main_model = joblib.load(file)
    result = main_model.predict(query)[0]
    if result == 0:
        st.markdown(f"<div align='center'><b>Final result of advertisement: <span style='color: red;'>"
                    f"Failure</span></b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div align='center'><b>Final result of advertisement: <span style='color: green;'>"
                    f"Success</span></b></div>", unsafe_allow_html=True)
        st.balloons()
