# NewsLens

NewsLens is a Python-based chatbot for news research. It fetches news articles from provided URLs, splits the text into smaller chunks, generates embeddings using OpenAI's language model, indexes them using FAISS for efficient retrieval, and utilizes the language model for question answering. Users can input queries, and NewsLens provides relevant answers along with the sources of the information, if available.

## Features
- Fetches news articles from URLs
- Splits text data into smaller chunks for processing
- Generates embeddings using OpenAI's language model
- Stores embeddings in a FAISS index for fast retrieval
- Utilizes OpenAI's language model for question answering
- Provides answers to user queries along with information sources

## Output
<img width="959" alt="image" src="https://github.com/yashkadam435/NewsLens-Efficient-News-Research-Tool/assets/108817280/b62164a9-9ce1-4dfe-b385-6a60f79fd60a">

## Technologies
- Python
- Streamlit
- OpenAI
- FAISS
  
## Installation
1.Clone this repository to your local machine using:

```bash
 git clone https://github.com/yashkadam435/NewsLens-Efficient-News-Research-Tool.git
```
2. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY=your_api_key_here
```
## Usage
1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.
  
- One can now ask a question and get the answer based on those news articles

## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.
