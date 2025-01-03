import streamlit as st
import pickle
import re
import nltk

nltk.download("punkt")
nltk.download("stopwords")

knn = pickle.load(open("knn.pkl","rb"))
tfidf = pickle.load(open("tfidf.pkl","rb"))


def cleanResume(resume_text):
    cleanText= re.sub(r"http\S+\s"," ",resume_text)
    cleanText= re.sub(r"RT|cc"," ",cleanText)
    cleanText= re.sub(r"#\S+\s"," ",cleanText)
    cleanText= re.sub(r"@\S+"," ",cleanText)
    cleanText= re.sub(r"[%s]" % re.escape("""!"#@$%^&*()_~:|;[]{}/?<>+=-""")," ",cleanText)
    cleanText= re.sub(r'[^\x00-\x7f]'," ",cleanText)
    cleanText= re.sub(r"\s+"," ",cleanText)
    return cleanText

#web app
def main():
    st.title("Resume Screening App")
    uploaded_file =st.file_uploader("Upload Resume",type=["pdf","txt"])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode("latin-1")

        cleaned_Resume= cleanResume(resume_text)
        input_features=tfidf.transform([cleaned_Resume])
        prediction_id = knn.predict(input_features)[0]
        st.write(prediction_id)

        category_mapping = {
                            15: "Java Developer",
                            23: "Testing",
                            8: "Devops Engineer",
                            20: "Python Developer",
                            24: "Web Designing",
                            12: "HR",
                            13: "Hadoop",
                            3: "Blockchain",
                            10: "ETL Developer",
                            18: "Operations Manager",
                            6: "Data Science",
                            22: "Sales",
                            16: "Mechanical Engineering",
                            1: "Arts",
                            7: "Database",
                            11: "Electrical Engineering",
                            14: "Health and Fitness",
                            19: "PMO",
                            4: "Business Analyst",
                            9: "DotNet Developer",
                            2: "Automation Testing",
                            17: "Network Security Engineer",
                            21: "SAP Developer",
                            5: "Civil Engineer",
                            0: "Advocate",
                            }
        category_name = category_mapping.get(prediction_id, "unknown")
        st.write("Predicted Category:", category_name)





#python main
if __name__=="__main__":
    main()
