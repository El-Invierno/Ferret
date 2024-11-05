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
from langchain_core.prompts import FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableLambda

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
        You are parsing a job description. Try to get all the important text data from the JD.
        Get all the important technical skills that are mandatory and the other skills that are optional as well.
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

def createChain():
    '''This method creates a chain for processing the resume and returns the chain.
       Inclusion of OneShotPrompting is done here to demonstrate expected ouput improvement.
       A CustomParser is used to get structured output.'''
    example_prompt = PromptTemplate.from_template(
        '''
        Name of Candidate:\n{candidate_name}\n\n
        Job Description(JD):\n{job_description}\n\n
        Years of Experience(related to the JD):\n{years}\n\n
        Candidate's Resume:\n{candidate_resume}\n\n
        Reasoning & Comparison of JD v/s Resume on a fixed set of rubrics followed by assigning the resume a score on the scale of 10:\n\n{evaluation_logic}\n\n
        '''
    )
    examples = [
        {
            'candidate_name' : 'Alice Johnson',
            'years' : '3 years',
            'job_description' : '''
Job Title: Generative AI Engineer
Location: [City, State or Remote]
Department: AI & Machine Learning
Reports To: Head of AI Engineering / Chief Technology Officer (CTO)

Job Overview:
We are seeking an experienced and highly motivated Generative AI Engineer to join our advanced AI and machine learning team. The ideal candidate will possess a deep understanding of machine learning principles, with a focus on generative AI models, including transformer-based models, diffusion models, GANs, and autoregressive networks. This role involves designing, implementing, and optimizing generative AI systems that push the boundaries of artificial intelligence capabilities and contribute to innovative product development.

Key Responsibilities:
Model Development & Optimization:

Design and develop generative AI models such as GPT, BERT, DALL-E, and custom transformer architectures to generate high-quality content, including text, images, and multimedia.
Fine-tune pre-trained models and develop novel architectures that meet specific product or client needs, with a focus on performance and scalability.
Conduct hyperparameter tuning, model distillation, and quantization to enhance model performance and efficiency.
Data Collection & Preprocessing:

Collaborate with data engineers and data scientists to source, preprocess, and augment large-scale datasets for training generative models.
Implement data augmentation and pre-processing techniques that support efficient and robust model training.
Ensure compliance with ethical AI guidelines, including privacy and bias mitigation, in data collection and model training.
Pipeline & Deployment:

Develop automated pipelines for data ingestion, model training, evaluation, and deployment using tools such as PyTorch Lightning, TensorFlow Extended (TFX), and MLFlow.
Deploy generative models into production environments, ensuring integration with existing infrastructure using Kubernetes, Docker, or cloud services (AWS, Azure, GCP).
Implement continuous monitoring and evaluation of models in production, leveraging feedback loops to improve accuracy and relevance.
Research & Innovation:

Stay up-to-date with advancements in generative AI and contribute to the company’s research initiatives.
Experiment with state-of-the-art models, including diffusion models, GANs, and autoregressive models, for both internal and external applications.
Collaborate with academia and industry leaders, contributing to research papers, patents, and whitepapers as part of the R&D process.
Product Collaboration:

Work closely with product managers, UX/UI designers, and software engineers to integrate generative AI capabilities into user-facing products.
Translate business requirements into technical solutions, ensuring that generative models meet product goals and user needs.
Provide insights and technical support during the ideation and prototyping phases of new AI-driven features and functionalities.
Ethical AI & Compliance:

Advocate for ethical AI practices, ensuring fairness, transparency, and accountability in generative model outputs.
Develop and enforce compliance with AI-related regulations, ensuring models adhere to standards for bias, privacy, and security.
Conduct regular audits of model performance and outputs to detect and mitigate biases or other unintended consequences.
Team Collaboration & Mentorship:

Collaborate with cross-functional teams to integrate generative AI models across various business functions.
Mentor junior machine learning engineers, providing guidance on best practices, troubleshooting, and model development.
Participate in code reviews and knowledge-sharing sessions to ensure high standards of coding, documentation, and reproducibility.
Qualifications:
Educational Background:

Bachelor’s or Master’s degree in Computer Science, Data Science, Mathematics, Statistics, or a related field. A Ph.D. is a plus.
Experience:

3+ years of experience in machine learning, with a focus on generative AI models such as transformers, GANs, diffusion models, or variational autoencoders.
Proven experience deploying large-scale machine learning models in production environments.
Technical Skills:

Programming: Proficiency in Python and ML libraries such as TensorFlow, PyTorch, and Hugging Face Transformers.
Modeling: Strong understanding of generative AI architectures, including transformers, GANs, diffusion models, VAEs, and autoregressive networks.
Data Management: Experience with data engineering tools like Apache Spark, SQL, and NoSQL databases.
Cloud Computing: Familiarity with cloud services (AWS, GCP, Azure) for deploying and scaling AI models.
Containerization & Deployment: Skilled in using Docker, Kubernetes, and CI/CD pipelines for model deployment.
Other Tools: Experience with MLFlow, DVC, and version control systems (Git).
Soft Skills:

Strong analytical and problem-solving skills with attention to detail.
Excellent communication skills for conveying complex technical information to both technical and non-technical stakeholders.
Ability to work collaboratively within a team environment and across departments.
Preferred Qualifications:

Familiarity with reinforcement learning and simulation environments for generative AI.
Background in ethical AI, explainable AI, or model interpretability.
Publications or contributions to research in the field of generative AI.''',

        'candidate_resume' : '''
Name: Alice Johnson
Contact Information:

Phone: +1 (555) 123-4567
Email: alice.johnson@gmail.com
LinkedIn: linkedin.com/in/alice-johnson-ai
Objective:
Experienced AI Engineer specializing in generative AI models and production-level machine learning. Passionate about leveraging transformer architectures and advanced deep learning techniques to drive impactful, scalable AI solutions in a collaborative environment.

Experience:

Senior AI Engineer
TechWave Solutions, San Francisco, CA
March 2020 – Present

Designed and deployed GPT-based models for customer support automation, achieving a 40% reduction in response time.
Developed and fine-tuned large transformer models for document summarization and question-answering tasks, handling over 1 million documents with 95% accuracy.
Collaborated with cross-functional teams to integrate generative AI solutions into web and mobile applications, ensuring seamless user experiences.
Conducted regular audits and fine-tuning sessions to mitigate bias in AI models, implementing ethical AI practices across projects.
Mentored a team of junior AI engineers, promoting best practices in model deployment, version control, and data privacy.
Machine Learning Engineer
DataFount AI Labs, Boston, MA
July 2017 – February 2020

Developed GAN models for image synthesis, improving visual quality by 30% and achieving a FID score reduction from 70 to 45.
Spearheaded a project using variational autoencoders (VAEs) for anomaly detection in financial transactions, reducing false positives by 25%.
Built and maintained continuous integration pipelines for ML models in production, utilizing Docker, Kubernetes, and AWS S3 for scalable deployments.
Technical Skills:

Programming: Python (TensorFlow, PyTorch), SQL
Generative Models: Transformers (BERT, GPT-3), GANs, Diffusion Models, VAEs
Data Engineering: Spark, SQL, NoSQL Databases
Cloud Platforms: AWS, GCP, Azure
Deployment: Docker, Kubernetes, CI/CD, MLFlow
Other Tools: DVC, Git, Apache Spark
Projects:

Automated Content Generation Platform
Developed a content generation tool using GPT-3 for automated content creation, significantly reducing the time required for content production by 60%. Conducted extensive model fine-tuning to ensure content relevance and coherence.

Fraud Detection Using VAEs
Built a variational autoencoder for detecting fraud in financial transactions. Achieved a 92% accuracy rate, reducing false positives and improving client trust.

Education:
Master of Science in Computer Science
Stanford University, Palo Alto, CA
Graduated: 2017

Bachelor of Science in Mathematics and Computer Science
University of California, Berkeley, CA
Graduated: 2015

Certifications:

Certified Generative AI Specialist – DeepLearning.ai, 2022
AWS Certified Solutions Architect – Associate, 2021
''',
'evaluation_logic' : '''
    1. **Experience Relevance to Role**: 
        - The candidate has 5 years of professional experience in data science and machine learning, specifically within roles that involve implementing and optimizing generative AI models. 
        - This experience includes direct work with transformer-based models like GPT, model deployment in production environments, and hands-on fine-tuning of models for specific applications.
        - Additionally, the candidate's experience with cross-functional collaboration supports the job description's emphasis on working alongside product managers and engineers.
        - **Score**: 9/10

    2. **Companies Worked For**: 
        - The candidate’s experience at TechWave Solutions and DataFount AI Labs adds significant credibility. Both companies are known for their focus on data science and advanced AI solutions, which directly aligns with the industry's cutting-edge advancements.
        - These reputable organizations provide a solid foundation for the candidate’s skills, particularly in applying generative AI in real-world settings and delivering projects that impact core business functions.
        - **Score**: 8/10

    3. **Projects Related to Job Description**:
        - The candidate's project experience includes developing a GPT-based content generation tool and an anomaly detection model using variational autoencoders (VAEs). Both projects demonstrate applied knowledge of generative AI and transformer models, which are highly relevant to the role.
        - The content generation tool is particularly aligned with the JD’s focus on generative models for high-quality text generation, while the anomaly detection project shows versatility and innovation in using generative models for other applications.
        - These projects also highlight the candidate’s ability to design and implement complex models with a focus on impact and scalability, which is a critical aspect of the job description.
        - **Score**: 8/10

    4. **Research Work Related to Job Skills**:
        - While the candidate’s resume does not explicitly mention published research or formal academic contributions, their experience in model optimization, ethical AI practices, and bias mitigation demonstrates applied research skills.
        - The job description emphasizes a preference for research contributions, so a lack of formal publications slightly lowers the score. However, the candidate’s hands-on expertise in ethical AI and model fine-tuning contributes positively toward this criterion.
        - **Score**: 6/10

    5. **Value Addition and Impact Metrics**:
        - The candidate has a notable achievement of reducing processing time by 30%, which illustrates a strong ability to optimize model performance and demonstrates clear value addition.
        - Additionally, their work in improving GAN models’ FID scores showcases a focus on quality improvements in generative models, which aligns with the JD’s requirements for continuous model enhancement and efficiency.
        - These measurable achievements underscore the candidate’s potential impact within a generative AI role, providing evidence of their contributions to previous projects.
        - **Score**: 9/10

    **Overall Resume Score**: 8/10
'''
        }
    ]
    
    initial_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix='JD:\n{JDinput}\n\nCandidate Resume:\n{CandidateInput}',
        input_variables=['JDinput','CandidateInput'],
    )

    extraction_prompt = PromptTemplate.from_template(
        '''
        The evaluation remarks and the score generated from the pre vious model are:
        {input}
        Now using these remarks and final score generate a python list of comma seperated values.
        Output should be structured as follows:
        
        [name_of_candidate:str, [skills]:list of str -> (Strictly include the skillset do not include anything else.), years of experience:int, final_score:float]

        Make sure to omit any kind of additional data or code or comments apart from the one shared in the output template.
'''
    )    
    evaluation_chain = (initial_prompt | model | StrOutputParser() | {'input' : RunnablePassthrough()} | extraction_prompt |
              model | StrOutputParser() | RunnableLambda(lambda x : re.sub(r"```python\s*|\s*```","",x))
            )
    return evaluation_chain

def executeChain(chain,jdPath,resumeDir):
    '''Method created to execute the chain for all the resumes and jd.'''
    jd_file_path = jdPath
    resume_dir = resumeDir

    result_list = []
    
    with open(jd_file_path,'r',encoding='utf-8') as jd_file:
        job_desc = jd_file.read()
    
    for resume_file_name in os.listdir(resume_dir):
        resume_file_path = os.path.join(resume_dir,resume_file_name)
        if os.path.isfile(resume_file_path):
            with open(resume_file_path,'r',encoding='utf-8',errors='ignore') as resume_file:
                candidateResume = resume_file.read()
            output = chain.invoke({
                'JDinput' : job_desc,
                'CandidateInput' : candidateResume
            })
            
            print(f"Results for {resume_file_name}:")
            try:
                output_list = ast.literal_eval(output)                
                print("Output List (as formatted string):")
                print(type(output_list))
                print(output_list)
                result_list.append(output_list)
            except (ValueError, SyntaxError) as e:
                print("Error parsing output:", e)
                print("Raw output:", output)

            print("\n" + "=" * 50 + "\n")
    return result_list

def topKReranker(results,k):
    """Select the top k results out of the result set and return."""
    print(f'The number of resumes that have been collected are: {len(results)}')
    sorted_list = sorted(results,key=lambda x : x[-1],reverse=True)
    return sorted_list[:k]


def main():
    '''Parsing the resumes.'''
    # documents = SimpleDirectoryReader(
    #     './jobData/Resumes/normal_data/ResumePdfs',
    #     file_extractor=file_extractor,
    # ).load_data()
    # mergedDict = mergeDocs(documents) # Merge the documents into a dict.
    # createFiles(mergedDict,'./jobData/Resumes/final_data/') # Create .txt files for each resume.

    '''Parsing the JD.'''
    jdDoc = SimpleDirectoryReader(
        './jobData/JDs/glean_jd',
        file_extractor=file_extractor2
    ).load_data()
    mergedJD = mergeDocs(jdDoc)
    createFiles(mergedJD,'./jobData/JDs/final_data/')

    # chain = createChain()
    # results = executeChain(chain,'./jobData/JDs/final_data/Glean_JD_Quality Assurance Engineer.txt','./jobData/Resumes/final_data/') # Load results.
    # topKReranker(results,10) # re-ranks and selects topK selects from the output.


if __name__ == "__main__":
    main()


    '''Execute comparisons of JD vs Resume:=
    
        Method 1: Use pure prompt engineering without RAG, only LLM calling and decision making for each document.
        Include a 1 shot prompt to show the model how to judge the resume, with correct input and output.
        Parse the structured output using Pydantic Classes and then output the data to the sql database. [Implemented]

        Method 2: Use RAG with metadata retrieval. [Not Feasible for small documents like resumes or jds]

    '''