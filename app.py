import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
import mysql.connector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from crewai import Agent, Task, Crew
import pandas as pd
from io import StringIO
import re
from datetime import datetime
import os
import plotly.express as px

# Initialize LLM and database connection
os.environ['GROQ_API_KEY'] = "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"
llm = ChatGroq(groq_api_key="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", model="llama3-70b-8192")
embedder = OllamaEmbeddings(model="nomic-embed-text")
my_db = mysql.connector.connect(host="localhost", user='root', passwd="naruto", database='test')
mycursor = my_db.cursor()

agent7 = Agent(
    role='User Performance Diagnostics Specialist',
    goal="""Analyze incorrect questions and identify specific knowledge gaps, learning weaknesses, and improvement areas for users across different topics.""",
    backstory="""You are an advanced diagnostic agent specialized in educational performance analysis. Your primary objective is to:
    - Systematically evaluate user's incorrect responses
    - Identify precise knowledge gaps""",
    llm="groq/llama3-70b-8192",
    max_iter=3,
    verbose=True,
    allow_delegation=False
)

def weak_areas(paragraph, topic):
    performance_task = Task(
        description=f"""Analyze incorrect questions {paragraph} for topic: {topic}
        
        Analysis Requirements:
        - Identify specific knowledge gaps
        - Highlight weak areas in the concept
        
        Diagnostic Dimensions:
        - Conceptual Weakness Mapping
        
        Output Format:
        - Structured markdown report
        - Quantitative performance metrics
        
        Key Focus Areas:
        - Pattern of misconceptions
        - Difficulty level correlation
        - Topic-specific challenge areas""",
        
        agent=agent7,
        expected_output="""Comprehensive performance diagnostic report with identifying all the highlighting all the weak areas"""
    )

    crew = Crew(
        agents=[agent7],
        tasks=[performance_task]
    )

    # Execute the task
    result = crew.kickoff()
 # Pretty-printed string
    return result.raw

agent8 = Agent(
    role='Personalized Learning Strategy Developer',
    goal='Generate targeted, adaptive learning improvement strategies based on individual performance analysis',
    backstory="""An AI-powered learning optimization specialist focused on:
    - Suggest topics to work on 
    - Suggest difficulty levels to work on 
    - Suggest type of questions to work on
    - Designing customized improvement pathways
    - Recommending strategic learning interventions
    - Transforming weaknesses into learning opportunities""",
    llm="groq/llama3-70b-8192",
    max_iter=3,
    verbose=True
)

def generate_improvement_strategy(paragraph, topic):
    improvement_task = Task(
        description=f"""Generate actionable learning improvement strategy for the mistakes in questions with difficulty {paragraph} from {topic}
        
        Strategy Development Requirements:
        - Propose targeted improvement actions
        - Create structured learning roadmap
        
        Recommendation Dimensions:
        1. Topic-specific skill enhancement
        2. Question type practice strategies
        3. Difficulty level progression
        4. Conceptual understanding reinforcement
        """,
        
        agent=agent8,
        expected_output="""Comprehensive improvement strategy with:
        - Precise learning recommendations
        - Targeted practice suggestions
        - Skill development roadmap"""
    )

    crew = Crew(
        agents=[agent8],
        tasks=[improvement_task]
    )

    # Execute the task
    result = crew.kickoff()  # Pretty-printed string
    return result.raw

agent9 = Agent(
   role='Performance Analysis and Skill Profiler',
   goal='Conduct comprehensive user performance evaluation by identifying strengths, weaknesses, and learning potential',
   
   backstory="""Advanced performance diagnostic specialist focused on:
   - Detailed answer pattern analysis
   - Skill competency mapping
   - Personalized performance insights
   - Constructive feedback generation""",

   
   llm="groq/llama3-70b-8192",
   max_iter=3,
   verbose=True
)

def analyze_user_performance(correct_answers, wrong_answers , topic):
   """
   Generate comprehensive performance analysis
   """
   performance_task = Task(
       description=f"""Analyze user performance across answers with correct questions with difficulty as {correct_answers} and wrong questions with difficulty as {wrong_answers} for {topic}
       
       Analysis Dimensions:
       - Strengths and Weaknesses of the user
       - Answer pattern recognition
       - Skill competency mapping
       - Strength and weakness identification
       - Learning potential evaluation""",
       
       agent=agent9,
       expected_output=f"Detailed performance diagnostic report including the strength and weaknesses of the user in the topic {topic}"
   )




   crew = Crew(
        agents=[agent9],
        tasks=[performance_task]
    )

        # Execute the task
   result = crew.kickoff()
   return result.raw

# Streamlit app starts here
st.title("RAG Test Application")

# Step 1: Upload Document
st.header("Upload Document for RAG Application")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    temp_file_path = os.path.join("temp_uploaded_file.pdf")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process document
    st.write("Processing the document...")
    loader = PyPDFLoader(temp_file_path)
    text = loader.load()

    os.remove(temp_file_path)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final = splitter.split_documents(text)

    # Create embeddings and FAISS database
    db = FAISS.from_documents(final, embedder)
    retriever = db.as_retriever()

    # Define prompt and RAG chains
    retriever_prompt = ("""You are a specialized question generator for CSV-formatted assessment for the {context}.

STRICT OUTPUT FORMAT:
- Begin response with first question immediately
- No headers, introductions, or explanatory text allowed
- Produce ONLY CSV content

Question Generation Rules:
1. Generate 10 unique questions about the topic
2. Each question must:
   - Be distinct in wording
   - Cover different aspects of the topic
   - Avoid repetition

CSV Column Requirements:
- question
- optionA
- optionB
- optionC
- optionD
- correct 
- difficulty

Question Characteristics:
- Test comprehensive understanding
- Include varied difficulty levels
- Cover multiple knowledge dimensions
- Reflect topic's critical aspects

Difficulty Distribution:
- 3 Easy questions
- 4 Moderate questions
- 3 Challenging questions

Formatting Restrictions:
- NO introductory phrases
- NO additional explanatory text
- Direct CSV output only
- One clear correct answer per question
                    
NO EXTRA TEXT OR HEADLINES OR ENDING LINES
DO NOT INCLUDE ANY HEADINGS OR EXTRA TEXTS
STRICTLT EXCLUDE ANY EXTRA TEXTS OTHER THAN THE CSV

Output Expectation:
Precise, structured CSV questions without any supplementary information""")
    
    prompt = ChatPromptTemplate.from_messages([("system", retriever_prompt), ("human", "{input}")])
    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    prompt_2 = ChatPromptTemplate.from_template(
    """Use ONLY the following context to answer the question. 
    If the answer is not in the context, say "I cannot find the answer in the provided document."

    Context:
    {context}

    Question: {input}

    """
)   
    
    retriever_2 = db.as_retriever()

    document_chain_2 = create_stuff_documents_chain(llm ,prompt_2 )

    chain_2 = create_retrieval_chain(retriever_2 , document_chain_2)

    inpu = "what is this document about in 5 words or less"

    response = chain_2.invoke({"input" : inpu})

    topic = response['answer']


    # Step 2: Generate Test
# Step 2: Generate Test
# Step 2: Generate Test
    st.header("Generate Questions")
    if "test_generated" not in st.session_state:
        st.session_state.test_generated = False  # Track if the test is generated

    if st.button("Generate Test") or st.session_state.test_generated:
        st.session_state.test_generated = True
        if "df" not in st.session_state:  # Avoid regenerating if already generated
            inpu = "Generate 10 questions from the file provided"
            response = chain.invoke({"input": inpu})
            csv_content = response['answer']
            cleaned_content = re.sub(r"^.*generated questions in CSV format.*\n?", "", csv_content, flags=re.MULTILINE)
            csv_data = StringIO(cleaned_content)
            st.session_state.df = pd.read_csv(csv_data)  # Save the DataFrame in session_state
        st.write("Test generated successfully!")

    # Step 3: Take Test
    if st.session_state.test_generated:
        st.header("Take the Test")
        df = st.session_state.df  # Access the saved DataFrame

        if "answers" not in st.session_state:
            st.session_state.answers = [None] * len(df)  # Placeholder for answers

        for i in range(df.shape[0]):
            st.write(f"Q{i+1}: {df['question'].iloc[i]}")

            # Display radio buttons for answers
            selected_answer = st.radio(
                f"Your Answer for Q{i+1} (e.g., A, B, C, or D):",
                options=[f"{df['optionA'].iloc[i]}", f"{df['optionB'].iloc[i]}", f"{df['optionC'].iloc[i]}", f"{df['optionD'].iloc[i]}"],
                key=f"q{i}",
                index=[f"{df['optionA'].iloc[i]}", f"{df['optionB'].iloc[i]}", f"{df['optionC'].iloc[i]}", f"{df['optionD'].iloc[i]}"].index(st.session_state.answers[i]) if st.session_state.answers[i] else 0,
            )

            # Update session_state with the selected answer
            st.session_state.answers[i] = selected_answer

        # Step 4: Evaluate Test
        if st.button("Submit Test"):
            df['given'] = st.session_state.answers
            wrong = []

            for i in range(df.shape[0]):
                if df['correct'].iloc[i] != df['given'].iloc[i]:
                    wrong.append("wrong")
                else:
                    wrong.append("correct")

            percentage = (wrong.count("correct") / df.shape[0]) * 100
            correct_count = wrong.count("correct")
            wrong_count = wrong.count("wrong")

            # Display results
            st.write(f"You scored {correct_count} out of {len(wrong)} ({percentage:.2f}%)")
            st.write("### Incorrect Questions:")
            for i in range(len(wrong)):
                if wrong[i] == "wrong":
                    st.write(f"Q: {df['question'].iloc[i]}")
                    st.write(f"Correct Answer: {df['correct'].iloc[i]}, Your Answer: {df['given'].iloc[i]}")

            # Save results in the database
            today = datetime.now().strftime("%Y-%m-%d")
            mycursor.execute(
                "INSERT INTO test_results (test_name, percentage, correct, wrong, date_of_test) VALUES (%s, %s, %s, %s, %s)",
                (topic, percentage, correct_count, wrong_count, today),
            )
            my_db.commit()


            # Modify the button to toggle the state
        if st.button("Analyze Weak Areas"):
            st.session_state.analyze_clicked = True

        if st.session_state.get("analyze_clicked", False):
            df['given'] = st.session_state.answers
            wrong = []

            for i in range(df.shape[0]):
                if df['correct'].iloc[i] != df['given'].iloc[i]:
                    wrong.append("wrong")
                else:
                    wrong.append("correct")
            paragraph = ". ".join([
                f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                for i in range(len(wrong)) if wrong[i] == 'wrong'
            ])
            if not paragraph.strip():
                st.error("No incorrect questions found to analyze.")
            else:
                with st.spinner("Analyzing weak areas..."):
                    try:
                        result = weak_areas(paragraph, topic)
                        st.success("Analysis Complete!")
                        st.write("Weak Area Analysis:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")


        # Generate Improvement Strategy Button
        if st.button("Generate Improvement Strategy"):
            st.session_state.strategy_clicked = True

        if st.session_state.get("strategy_clicked", False):
            paragraph = ". ".join([
                f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                for i in range(len(wrong)) if wrong[i] == 'wrong'
            ])
            if not paragraph.strip():
                st.error("No incorrect questions found to generate strategy.")
            else:
                with st.spinner("Generating improvement strategy..."):
                    try:
                        result = generate_improvement_strategy(paragraph, topic)
                        st.success("Strategy Generated!")
                        st.write("Improvement Strategy:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error during strategy generation: {e}")

        if st.button("Weak and Strength"):
            st.session_state.wc_clicked = True

        if st.session_state.get("wc_clicked", False):
            paragraph = ". ".join([
                f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                for i in range(len(wrong)) if wrong[i] == 'wrong'
            ])
             
            right= ". ".join([
                f"{df['question'].iloc[i]} (Difficulty: {df['difficulty'].iloc[i]})"
                for i in range(len(wrong)) if wrong[i] == 'correct'
            ])

            if not paragraph.strip():
                st.error("No incorrect questions found to generate weakness.")
            else:
                with st.spinner("Generating improvement strategy..."):
                    try:
                        result = analyze_user_performance(right , paragraph, topic)
                        st.success("Strength/Weakness Generated!")
                        st.write("Strength/Weakness:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error during strategy generation: {e}")

        if st.button("Analysis From Past Tests"):
            st.session_state.past_clicked = True

        if st.session_state.get("past_clicked", False):
            mycursor.execute('SELECT * FROM test_results ')
            z = mycursor.fetchall()

            # Get column names
            column_names = [i[0] for i in mycursor.description]

            # Create a DataFrame
            sql = pd.DataFrame(z, columns=column_names)
            
            with st.spinner("Generating Analysis..."):
                try:
                    average_score = sql['percentage'].mean()
                    st.write(f"Your average score from previous tests are {average_score}")
                    sql['date_of_test'] = pd.to_datetime(sql['date_of_test'])
                    fig = px.line(sql, x='date_of_test', y='percentage', title='Line Graph of Date vs Test Percentage', markers=True)
                    fig.update_layout(xaxis_title="Date (yyyy-mm-dd)", yaxis_title="Percentage")

                    # Display the graph in Streamlit
                    st.write("Line Graph of Percentage with Plotly")
                    st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error during analysis generation: {e}")


