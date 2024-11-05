import os
import re
import ast
from pydoc import doc
from nt import environ
from dotenv import load_dotenv
load_dotenv()
from genericpath import isfile
from distutils.util import execute
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

parser1 = LlamaParse(
    parsing_instruction='''
        You are parsing a resume. Try to get all the important text data from the resume.
        Please do not parse useless items such as hobbies, human languages known, signature, and if any images, clipart or unnecessary designs are present in the resume.
        You are free to ignore the useless sections in the resume.
    ''',
    show_progress=True,
) # Initialize an instance of LlamaParse.
file_extractor1 = {'.pdf': parser1}

parser2 = LlamaParse(
    parsing_instruction='''
        You are parsing a job description. Try to get all the important text data from the JobDescription.
    ''',
    show_progress=True,
) # Initialize an instance of LlamaParse.
file_extractor2 = {'.pdf': parser1}

model = ChatOpenAI(model='gpt-4o',temperature=0) # ChatOpenAI model initialization.

def mergeDocs(documents):
    '''This function is used to merge the Document objects with the same filename metadata.'''

    documents_dict = {}
    for doc in documents:
        file_name = doc.metadata['file_name'].split('.')[0] + '.txt'
        if file_name in documents_dict:
            documents_dict[file_name] += '\n' + doc.text
        else:
            documents_dict[file_name] = doc.text
    return documents_dict

def createFiles(mergedDict, outdir):
    '''Creates .txt files for each resume using parsed data stored in the dict.'''

    output_dir = outdir
    os.makedirs(output_dir, exist_ok=True)
    for file_name, text in mergedDict.items():
        file_path = os.path.join(output_dir,file_name)
        with open(file_path,'w',encoding='utf-8') as file:
            file.write(text)
        print(f'The output has been written to the file path {file_path}')

def compareEvaluate():
    # First let us summarize and extract important info from the JobDesc.
    job_desc_path = './jobData/JDs/final_data/Glean_JD_Quality Assurance Engineer.txt'
    with open(job_desc_path,'r',encoding='utf-8',errors='ignore') as job_file:
        job_desc = job_file.read()

    result_list = []
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
    jdOutput = jdChain.invoke({'jobDescription' : job_desc})

    # Next let us summarize and extract important info from the Resumes.
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
    resume_dir = './jobData/Resumes/final_data/'

    comparison_prompt = PromptTemplate.from_template(
        '''
            <Evaluation of the resume content with reference to the job description.>
            Job Description: {Job_Desc}
            Resume Description: {Resume_Desc}
            - Evaluate the resume description using the job description as a point of reference. 
            - The final resume evaluation score should be out of 10. The score can be a float value.
            - If years of experience of candidate is more than or equal to the value specified then give high preference.
            - If most of the mandatory skills offered by the candidate matches the skillset demanded by the job description, then give higher score.
            - Check out the keywords from the experience and project sections of the resume and try to map them to different skills. The higher number of maps the higher the score.
            - If a candidate has performed research on a certain topic, then give more preference.
            - Similarly, awards and recognition are given the due weightage(if any), given that the awards are pertitnent to the skillset in demand.
            - Score the content on the above fixed set of Rubrics.
            * Do not hallucinate and use random data.
            Output(Python List):
            [<candidate_name>, <candidate_email>, [<matched_skillsets>], <score_out_of_10>]
            ** Do not output anything apart from the above output format.
        '''
    )
    comparison_chain = comparison_prompt | model | StrOutputParser() | RunnableLambda(lambda x : re.sub(r"```python\s*|\s*```","",x))

    for resume_file_name in os.listdir(resume_dir):
        resume_file_path = os.path.join(resume_dir,resume_file_name)
        with open(resume_file_path,'r',encoding='utf-8',errors='ignore') as resumeFile:
            resume_content = resumeFile.read()
        resumeOutput = resume_chain.invoke({'resumeDescription' : resume_content})
        intermediate_output = comparison_chain.invoke({'Job_Desc' : jdOutput, 'Resume_Desc' : resumeOutput})
        print(type(intermediate_output))
        print(intermediate_output)
        print('\n\n=================\n\n')
        output_list = ast.literal_eval(intermediate_output)
        result_list.append(output_list)
    return result_list


def topKReranker(results,k):
    """Select the top k results out of the result set and return."""
    print(f'The number of resumes that have been collected are: {len(results)}')
    sorted_list = sorted(results,key=lambda x : x[-1],reverse=True)
    return sorted_list[:k]


def main():
    '''Parsing the resumes.'''

    documents = SimpleDirectoryReader(
        './jobData/Resumes/normal_data/ResumePdfs',
        file_extractor=file_extractor1,
    ).load_data()
    mergedDict = mergeDocs(documents) # Merge the documents into a dict.
    createFiles(mergedDict,'./jobData/Resumes/final_data/') # Create .txt files for each resume.

    '''Parsing the JD.'''
    jdDoc = SimpleDirectoryReader(
        './jobData/JDs/glean_jd',
        file_extractor=file_extractor2
    ).load_data()
    mergedJD = mergeDocs(jdDoc)
    createFiles(mergedJD,'./jobData/JDs/final_data/')

    results = compareEvaluate()
    final_results = topKReranker(results,10)
    print(final_results)

if __name__ == "__main__":
    main()


    '''Execute comparisons of JD vs Resume:=
    
        Method 1: Use pure prompt engineering without RAG, only LLM calling and decision making for each document.
        Include a 1 shot prompt to show the model how to judge the resume, with correct input and output.
        Parse the structured output using Pydantic Classes and then output the data to the sql database. [Implemented]

        Method 2: Use RAG with metadata retrieval. [Not Feasible for small documents like resumes or jds]

    '''