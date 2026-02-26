import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from dotenv import load_dotenv

# Swapped OpenAI/Gemini for Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class TitanicChatAgent:
    """
    Main agent class that handles Titanic dataset queries
    """
    
    def __init__(self):
        """Initialize the agent with dataset and LLM"""
        # Load Titanic dataset from seaborn
        self.df = sns.load_dataset('titanic')
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", # Incredible logic model for coding
            temperature=0,             
            api_key=os.getenv("GROQ_API_KEY") 
        )
        
        # Create pandas dataframe agent
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            agent_type="tool-calling",  
            verbose=True,
            allow_dangerous_code=True
        )
        
        print("✅ Titanic Agent initialized successfully!")
        print(f"📊 Dataset loaded: {len(self.df)} passengers")
    
    def query(self, question: str):
        try:
            # Inject strict instructions behind the scenes!
            strict_prompt = (
                f"{question}\n\n"
                "IMPORTANT INSTRUCTION: You must calculate the answer using the pre-loaded dataframe named 'df'. "
                "Do NOT create a sample dataframe. Do NOT hallucinate data. Give me the final real answer."
            )
            
            # Pass the strict prompt instead of the raw question
            result = self.agent.invoke({"input": strict_prompt})

            if isinstance(result, dict):
                text_answer = result.get('output') or result.get('final_output') or str(result)
            else:
                text_answer = str(result)

            visualization = None
            viz_keywords = ['histogram', 'chart', 'plot', 'graph', 'show', 'visualize', 'distribution']
            if any(keyword in question.lower() for keyword in viz_keywords):
                visualization = self._create_visualization(question)

            return {
                "answer": text_answer,
                "visualization": visualization,
                "error": None
            }

        except Exception as e:
            return {
                "answer": None,
                "visualization": None,
                "error": f"Error processing query: {str(e)}"
            }
    
    def _create_visualization(self, question: str):
        """
        Create appropriate visualization based on question
        
        Args:
            question: User's query
            
        Returns:
            str: Base64 encoded image
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Detect what to visualize based on keywords
            q_lower = question.lower()
            
            if 'age' in q_lower:
                # Age histogram
                self.df['age'].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_xlabel('Age')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Passenger Ages')
                
            elif 'fare' in q_lower or 'ticket' in q_lower:
                # Fare histogram
                self.df['fare'].dropna().hist(bins=30, ax=ax, color='lightgreen', edgecolor='black')
                ax.set_xlabel('Fare')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Ticket Fares')
                
            elif 'sex' in q_lower or 'gender' in q_lower or 'male' in q_lower or 'female' in q_lower:
                # Gender distribution
                self.df['sex'].value_counts().plot(kind='bar', ax=ax, color=['lightblue', 'pink'])
                ax.set_xlabel('Gender')
                ax.set_ylabel('Count')
                ax.set_title('Passenger Gender Distribution')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                
            elif 'embark' in q_lower or 'port' in q_lower:
                # Embarkation port distribution
                self.df['embark_town'].value_counts().plot(kind='bar', ax=ax, color='coral')
                ax.set_xlabel('Embarkation Port')
                ax.set_ylabel('Count')
                ax.set_title('Passengers by Embarkation Port')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                
            elif 'class' in q_lower:
                # Passenger class distribution
                self.df['class'].value_counts().plot(kind='bar', ax=ax, color='lightseagreen')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.set_title('Passenger Class Distribution')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                
            elif 'surviv' in q_lower:
                # Survival distribution
                survival_counts = self.df['survived'].value_counts()
                survival_counts.index = ['Died', 'Survived']
                survival_counts.plot(kind='bar', ax=ax, color=['red', 'green'])
                ax.set_xlabel('Outcome')
                ax.set_ylabel('Count')
                ax.set_title('Survival Distribution')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                
            else:
                # Default: Age distribution
                self.df['age'].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_xlabel('Age')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Passenger Ages')
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
            return None
    
    def get_dataset_info(self):
        """Get basic info about the dataset"""
        return {
            "total_passengers": len(self.df),
            "columns": list(self.df.columns),
            "sample_data": self.df.head(3).to_dict('records')
        }

# Test the agent (only runs if you execute this file directly)
if __name__ == "__main__":
    print("🚀 Testing Titanic Chat Agent...")
    
    agent = TitanicChatAgent()
    
    # Test query
   # Test query with stricter instructions
    test_question = "Using the provided dataframe 'df', calculate the exact percentage of passengers who were male. Give me the final real number. Show me a chart."
    print(f"\n❓ Question: {test_question}")
    
    result = agent.query(test_question)
    
    if result['error']:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Answer: {result['answer']}")
        if result['visualization']:
            print("📊 Visualization generated successfully! (Base64 string ready for frontend render)")