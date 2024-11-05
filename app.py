import os
import re
import ast
import streamlit as st
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import tempfile

# Load environment variables
load_dotenv()

# Initialize model
model = ChatOpenAI(model='gpt-4o', temperature=0)

# Define the LlamaParse instances with the appropriate parsing instructions
resume_parser = LlamaParse(
    parsing_instruction='''
        You are parsing a resume. Try to get all the important text data from the resume.
        Please do not parse useless items such as hobbies, human languages known, signature, and if any images, clipart or unnecessary designs are present in the resume.
        You are free to ignore the useless sections in the resume.
    ''',
    show_progress=True,
)

jd_parser = LlamaParse(
    parsing_instruction='''
        You are parsing a job description. Try to get all the important text data from the JobDescription.
    ''',
    show_progress=True,
)

# File extractor for each file type
file_extractor = {'.pdf': resume_parser}
jd_extractor = {'.pdf': jd_parser}

# Define helper functions
def mergeDocs(documents):
    documents_dict = {}
    for doc in documents:
        file_name = doc.metadata['file_name'].split('.')[0] + '.txt'
        if file_name in documents_dict:
            documents_dict[file_name] += '\n' + doc.text
        else:
            documents_dict[file_name] = doc.text
    return documents_dict

def compareEvaluate(jd_content, resume_contents):
    summarizeKeywordPrompt = PromptTemplate.from_template(
        '''
            <Job-Description Context>:
            {jobDescription}\n
            Mandatory Points to follow:
            - Understand the job role and what is the educational background required to be eligible for it.
            - Try to narrow down on any minimum percentage or gradepoint criteria for the educational courses.
            - Extract all the key technical keywords and skills required in the job posting.
            - Extract the number of years of experience required from the description.
            - Try to understand all the skillset mentioned, and then classify it under mandatory skills and good to have.
            - Encompass all of the above points and extensively summarize the job description. Try to keep the summary information dense.
            - Summary should be in pointers to make it comprehensive.
            * Do not hallucinate and use random data.
        '''
    )
    jdChain = summarizeKeywordPrompt | model | StrOutputParser()
    jdOutput = jdChain.invoke({'jobDescription': jd_content})

    summarizeResumePrompt = PromptTemplate.from_template(
        '''
            <Resume Context>:
            {resumeDescription}\n
            Mandatory Points to follow:
            - Extract the personal details of the candidate ie. EmailId, Name.
            - Extract the technical skills mentioned in the Skills section of the resume.
            - Extract the years of experience that the candidate has in the industry.
            - Summarize the projects and extract important keywords from the project section.
            - Capture the metrics in percentage or just plain numbers in the project or experience section.
            - Draw out connections between standard industry skills and the keywords extracted from the experience or projects section.
            - Extract the important keywords from certifications if present in the resume. 
            - Encompass all of the above points and extensively summarize the resume. Keep the summary information dense, and retain important facts.
            - Summary should be in pointers to make it comprehensive.
            * Do not hallucinate and use random data.
        '''
    )
    resume_chain = summarizeResumePrompt | model | StrOutputParser()
    comparison_prompt = PromptTemplate.from_template(
        '''
            <Evaluation of the resume content with reference to the job description.>
            Job Description: {Job_Desc}
            Resume Description: {Resume_Desc}
            - Evaluate the resume description using the job description as a point of reference. 
            - The final resume evaluation score should be out of 10. The score can be a float value.
            - Score the content on the above fixed set of Rubrics.
            Output(Python List):
            [<candidate_name>, <candidate_email>, [<matched_skillsets>], <score_out_of_10>]
            ** Do not output anything apart from the above output format.
        '''
    )
    comparison_chain = comparison_prompt | model | StrOutputParser() | RunnableLambda(lambda x : re.sub(r"```python\s*|\s*```", "", x))

    results = []
    for resume_content in resume_contents:
        resumeOutput = resume_chain.invoke({'resumeDescription': resume_content})
        intermediate_output = comparison_chain.invoke({'Job_Desc': jdOutput, 'Resume_Desc': resumeOutput})
        output_list = ast.literal_eval(intermediate_output)
        results.append(output_list)
    
    return results

def topKReranker(results, k):
    sorted_list = sorted(results, key=lambda x: x[-1], reverse=True)
    return sorted_list[:k]

# Streamlit UI
col1, col2 = st.columns([1, 8])

with col1:
    st.image("./images/screenshot.jpeg", width=50)  # Adjust width as needed

with col2:
    st.markdown("<h1 style='display: inline;'>Ferret: Resume and Job Description Evaluation System</h1>", unsafe_allow_html=True)

st.header("Upload Job Description (PDF)")
jd_file = st.file_uploader("Choose a job description file", type=['pdf'])
jd_content = None

if jd_file:
    # Create a temporary directory for the JD PDF
    with tempfile.TemporaryDirectory() as tmp_jd_dir:
        # Save JD PDF in the temporary directory
        jd_path = os.path.join(tmp_jd_dir, "job_description.pdf")
        with open(jd_path, 'wb') as f:
            f.write(jd_file.read())

        # Use SimpleDirectoryReader to read the directory containing the JD PDF
        jd_documents = SimpleDirectoryReader(tmp_jd_dir, file_extractor=jd_extractor).load_data()
        jd_content = mergeDocs(jd_documents).get("job_description.txt", "")

st.header("Upload Resumes (PDFs)")
resume_files = st.file_uploader("Choose resume files", type=['pdf'], accept_multiple_files=True)

temp_dir = tempfile.mkdtemp()  # Create a temporary directory for resumes

# Save uploaded resumes to the temporary directory
for resume_file in resume_files:
    with open(os.path.join(temp_dir, resume_file.name), 'wb') as f:
        f.write(resume_file.read())

k = st.number_input("Enter the number of top candidates (K)", min_value=1, value=5)

if st.button("Evaluate"):
    if jd_content and resume_files:
        # Use SimpleDirectoryReader to read resumes from the temporary directory
        documents = SimpleDirectoryReader(temp_dir, file_extractor=file_extractor).load_data()
        mergedDict = mergeDocs(documents)

        # Compare and evaluate based on job description
        results = compareEvaluate(jd_content, list(mergedDict.values()))
        top_candidates = topKReranker(results, k)
        
        st.subheader("Top Candidates")
        for candidate in top_candidates:
            st.write(f"Name: {candidate[0]}")
            st.write(f"Email: {candidate[1]}")
            st.write(f"Matched Skills: {', '.join(candidate[2])}")
            st.write(f"Score: {candidate[3]}")
            st.write("---")
    else:
        st.error("Please upload both a job description and resumes.")
