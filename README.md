# FDA Cosmetic Guidance Q&A Chatbot

## Overview
This Streamlit app is a **Generative AI-powered chatbot** designed to answer questions related to FDA Cosmetic Guidance documents. It leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware responses grounded in the FDA's regulatory framework. The app is hosted on **Streamlit Cloud** and is part of the **PrecisionFDA Generative AI Community Challenge**.

**App Link**: [https://precisionchallenge-afqskmcjfbtscms4yfsgm7.streamlit.app/](https://precisionchallenge-afqskmcjfbtscms4yfsgm7.streamlit.app/)

---

## Features
1. **Regulatory Q&A**:
   - Ask questions about FDA Cosmetic Guidance documents and receive accurate, citation-backed answers.
2. **Source Citations**:
   - View the specific sections of the FDA Cosmetic Guidance document used to generate each response.
3. **Chat Memory**:
   - Maintains a history of questions and answers for context-aware conversations.
4. **Customizable Settings**:
   - Adjust parameters like **LLM temperature**, **document chunk size**, and **chunk overlap** for tailored responses.
5. **User-Friendly Interface**:
   - Simple and intuitive design for easy navigation and interaction.

---

## How to Use the App
1. **Access the App**:
   - Visit the app link: [https://precisionchallenge-afqskmcjfbtscms4yfsgm7.streamlit.app/](https://precisionchallenge-afqskmcjfbtscms4yfsgm7.streamlit.app/).
2. **Ask a Question**:
   - Enter your question in the text input box and press `Enter`.
3. **View the Response**:
   - The app will generate an answer based on the FDA Cosmetic Guidance document.
   - Relevant citations from the document will be displayed below the answer.
4. **Adjust Settings**:
   - Use the sidebar to customize the chatbot's behavior:
     - **LLM Temperature**: Controls the creativity of responses (0.0 for factual, 1.0 for creative).
     - **Document Chunk Size**: Adjusts the size of text chunks processed by the app.
     - **Chunk Overlap**: Controls the overlap between text chunks for better context.
5. **View Chat History**:
   - The sidebar displays a history of your questions and answers.
   - Use the **Clear Chat History** button to reset the conversation.

---

## Technical Details
### Technologies Used
- **Streamlit**: For building and hosting the web app.
- **LangChain**: For implementing the RAG pipeline.
- **OpenAI GPT-4**: For generating responses using a fine-tuned language model.
- **FAISS**: For efficient document retrieval and vector storage.
- **PyPDF**: For loading and processing the FDA Cosmetic Guidance PDF document.

### How It Works
1. **Document Processing**:
   - The FDA Cosmetic Guidance document is loaded and split into manageable chunks.
   - These chunks are embedded into a vector space using **OpenAI embeddings**.
2. **Retrieval-Augmented Generation (RAG)**:
   - When a user asks a question, the app retrieves the most relevant document chunks using **FAISS**.
   - The retrieved chunks are passed to the **OpenAI GPT-4 model** to generate a context-aware response.
3. **Response Generation**:
   - The app displays the generated answer along with citations from the retrieved document chunks.
