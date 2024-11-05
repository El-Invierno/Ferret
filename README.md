Goal of the Project The goal of Ferret is to automate the process of parsing resumes and job descriptions, extracting relevant information, and evaluating how well each resume matches a given job description. This system is intended to assist recruiters in identifying top candidates by analyzing essential skills, experience, and qualifications through a streamlined, automated process.

Key Components
Chains and Prompts with LangChain:

Job Description Parsing: Using a LangChain prompt template, Ferret analyzes job descriptions to extract critical requirements such as educational background, years of experience, and essential skills. The extracted information is summarized into structured points, making it easier to use as a reference for evaluating resumes.
Resume Parsing: A separate LangChain prompt template is applied to parse resumes. The system extracts information like the candidate's name, email, skills, years of experience, and notable project metrics. This output is structured for efficient comparison against job requirements.
Comparison Chain: Another LangChain prompt template is used to compare parsed resume data with job description data. The system evaluates key areas such as skills match, years of experience, and relevant achievements, generating a score out of 10 based on a fixed rubric.
Multithreading Optimization:

To handle large volumes of resumes efficiently, Ferret uses multithreading via ThreadPoolExecutor, allowing concurrent processing of resumes. This significantly speeds up the comparison and evaluation process.
Evaluation and Ranking:

After extracting and comparing data, the system ranks candidates by their scores and selects the top candidates who best match the job description. This list is then output, making it easy for recruiters to view the best-matched candidates.
Key Technologies
LangChain: Enables prompt engineering and chaining tasks, connecting language model outputs across multiple stages of parsing and comparison.
LlamaParse: Parses resumes and job descriptions to focus on the critical data points needed for evaluation.
OpenAI: Provides the language model for generating meaningful comparisons and extracting structured data.
Multithreading: Used to improve performance when handling multiple resumes simultaneously.
Outcome
By automating resume parsing, job description analysis, and candidate evaluation, Ferret helps recruiters quickly find top matches, streamlining the hiring process and reducing the manual effort typically involved in shortlisting candidates.
