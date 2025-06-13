import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Updated imports to fix deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

print("🤖 Initializing GPT-4.1 mini via OpenAI API...")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
print("✅ GPT-4.1 mini is ready!")

def load_pdf():
    pdf_name = 'hotemate.pdf'
    
    # Check if PDF file exists
    if not os.path.exists(pdf_name):
        raise FileNotFoundError(f"PDF file '{pdf_name}' not found in current directory")
    
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    ).from_loaders(loaders)
    return index

def main():
    try:
        print("🔄 Loading PDF and creating embeddings...")
        index = load_pdf()
        print("✅ PDF loaded successfully!")
        # Create QA chain with better prompt template
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    except FileNotFoundError as e:
        print(f"❌ PDF file error: {e}")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")

    if 'chain' not in locals():
        print("Sorry, the application is not properly configured. Please check the API key and PDF file.")
        return

    print("🤖 HotelMate AI")
    print("How can I assist you today? (Type 'exit' to quit)")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        if not prompt:
            continue

        try:
            # Use PDF knowledge for response
            result = chain.invoke({"query": prompt})
            response = result['result']

            # Clean up the response to make it more natural
            for prefix in ["According to the provided context, ", "According to the context, "]:
                if response.startswith(prefix):
                    response = response[len(prefix):]

            # Make first letter uppercase if it's lowercase after cleaning
            if response and response[0].islower():
                response = response[0].upper() + response[1:]

            print(f"HotelMate AI: {response}")

        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            # More specific error messages
            if "401" in str(e) or "invalid_api_key" in str(e):
                print("🔑 API Key Error: Your API key stopped working. Please check your OpenAI account.")
            elif "429" in str(e):
                print("⏰ Rate Limit: Too many requests. Please wait a moment and try again.")
            elif "insufficient_quota" in str(e):
                print("💳 Quota Error: You've exceeded your OpenAI usage quota.")
            else:
                print("I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question.")

if __name__ == "__main__":
    main()