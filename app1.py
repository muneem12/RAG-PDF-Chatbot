


import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from huggingface_hub import InferenceClient
from openai import OpenAI
import os
import time
from pypdf import PdfReader

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚"
)

st.title("📚 RAG PDF Chatbot")
st.write("Upload a PDF and ask questions about its content.")

#step12:add chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#step4:add the pdf upload interface
#pdf upload
uploaded_files = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    accept_multiple_files=True
)


#step6:split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


#step7:load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


#step9:retrieve relevant chunks
def retrieve_chunks(query, index, all_chunks, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    #normalize query embeddinng
    faiss.normalize_L2(query_embedding)

    # Search for relevant chunks
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)
    st.write("Simmilarity scores:", scores[0])
    results = []

    for score, idx in zip(scores[0], indices[0]):
        #skip invalid indices
        if idx < 0:
            continue
        if score < 0.15:
            continue  

      
        results.append({
            "filename": all_chunks[idx]["filename"],
            "page": all_chunks[idx]["page"],
            "text": all_chunks[idx]["text"],
            "score": float(score) 
        })
    if not results:
        st.warning(
            "No sufficiently relevant information was found in the PDF."
        )
        return []

    return results


# Step 1: Rewrite follow-up questions into standalone search questions
def rewrite_followup_question(query, api_key):
    # No history means this is the first question
    if not st.session_state.chat_history:
        return query

    # Use the latest 4 conversation exchanges
    recent_history = st.session_state.chat_history[-4:]

    history_text = "\n".join(
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in recent_history
    )

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the latest user question into a standalone "
                        "search query using conversation history to resolve "
                        "references such as it, they, this, and that. "
                        "Do not answer the question. Keep the original meaning. "
                        "Return only the rewritten question."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_text}\n\n"
                        f"Latest question: {query}"
                    )
                }
            ],
            temperature=0,
            max_tokens=100
        )

        rewritten_query = response.choices[0].message.content

        if rewritten_query and rewritten_query.strip():
            return rewritten_query.strip()

        return query

    except Exception as e:
        print("Question rewriting failed:", e)
        return query


#step10:generate an answer using Qwen
def generate_answer(query, results, api_key):
    context = "\n\n".join(
        f"Page {item['page']}:\n{item['text']}"
        for item in results
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a PDF question-answering assistant. "
                "Answer using only facts explicitly stated in the provided PDF context. "
                "Do not use outside knowledge or guess missing facts. "
                "Give a complete, direct answer in 2-4 sentences when appropriate. "
                "For multi-part questions, answer every part. "
                "Strictly return answers from PDF only. "
                "If asked about FAISS, answer only if the provided PDF context supports it. "
                "A topic mention or a question listed in the PDF is not an answer. "
                "If the answer is absent, say: "
                "'I could not find the answer in the provided PDF context.' "
                "Only cite information supported by relevant context. "
                "RAG can reduce unsupported answers but cannot guarantee correctness. "
                "Explain limitations when mentioned in the context. "
                "Do not output safety labels, status messages, or internal reasoning."
            )
                
            
        }
    ]
    #add previous conversation messages
    for item in st.session_state.chat_history:
        messages.append({
            "role": "user",
            "content": item["question"]
        })
        messages.append({
            "role": "assistant",
            "content": item["answer"]
        })

    # Add the current question and retrieved PDF context
    messages.append({
        "role": "user",
         "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    #try the more reliable model first, then use fallback
    models = [
        "openrouter/free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3.5-lightning:free",
        "liquid/lfm-2.5-2.6b:free"

        
        
    ]

    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0
            )

            #debugging information
            if not response.choices:
                st.warning(f"{model} returned no responses choices.")
                continue
            choice = response.choices[0]
            answer = (choice.message.content or "").strip()
            
            #debugging information
            st.write("Model used:", response.model)
            st.write("Finish reason:", choice.finish_reason)
            if not answer or not answer.strip():
                st.warning(f"{model} returned empty content.")
                continue

           

            #reject unexpected safety/status-only responses
            if answer.lower() in [
                "user safety: safe",
                "safe",
                "user safety: unsafe"
            ]:
                st.warning(f"{model} returned a status-only response.")
                continue
            
            #reject responses that appear truncated
            if choice.finish_reason == "length":
              st.warning(f"{model} returned a potentially incomplete answer.")
              
            return answer

        except Exception as e:
            #stop retrying when the API rate limit is reached
            if getattr(e, "status_code", None) == 429:
                return (
                    "OpenRouter free-model limit reached. "
                    "Please wait for the daily reset or add credits."
                )
            st.warning(f"Model {model} failed: {type(e).__name__}: {e}")
            print("Model error:", repr(e))
            time.sleep(1)
    return (
        "I couldn't generate an answer right now. "
        "Please try again."
    )


#step16:build and cache the FAISS index
@st.cache_resource
def build_faiss_index(texts):
    embedding_model = load_embedding_model()

    embeddings = embedding_model.encode(list(texts))

    #convert embeddings to float32
    embeddings = np.array(embeddings).astype("float32")

    #normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    #create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])

    #store embeddings
    index.add(embeddings)

    return index, embeddings


#process the uploaded PDF
if uploaded_files:

    #step5:extract text from multiple pdf
    all_chunks = []
    pages = []
    
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
               pages.append({
                   "page": page_num,
                   "filename": uploaded_file.name,
                   "text": text
               })
               #split text into chunks
               chunks = split_text(text, chunk_size=500, overlap=50)
       
               for chunk in chunks:
                   all_chunks.append({
                       "page": page_num,
                       "filename": uploaded_file.name,
                       "text": chunk
                   })
    
    st.write("Total PDFs uploaded:", len(uploaded_files))
    st.write("Total pages with text:", len(pages))
    st.write("Total text chunks:", len(all_chunks))
    if pages:
        st.subheader("Extracted Text Preview")
        st.write("File:", pages[0]["filename"])
        st.write("Page number:", pages[0]["page"])
        st.write(pages[0]["text"][:1000])
    else:
        st.warning("No extractable text found in the uploaded PDFs.")
    if all_chunks:

        #load the embedding model
        embedding_model = load_embedding_model()

        #generate embeddings
        texts = [chunk["text"] for chunk in all_chunks]
        #clear cached FAISS INdexes
        build_faiss_index.clear()
        with st.spinner("Generating embeddings..."):
            index, embeddings = build_faiss_index(tuple(texts))

        st.success("FAISS database created!")
        st.write("Total vectors:", index.ntotal)
        st.write("Embeddings generated successfully!")
        st.write("Embedding shape:", embeddings.shape)

        #step11:add a question input to streamlit
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""

        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            key="api_key"
        )

        #question input
        # Question input
        if "question" not in st.session_state:
            st.session_state.question = ""

        def clear_question():
            st.session_state.question = ""

        query = st.text_input(
            "Ask a question about your PDF",
            key="question"
        )

        col1, col2 = st.columns(2)

        with col1:
            get_answer = st.button("Get Answer")

        with col2:
            st.button("🧹 Clear Question", on_click=clear_question)

        if get_answer:
            if not query.strip():
                st.warning("Please enter a question.")

            elif not api_key:
                st.warning("Enter your OpenRouter API Key.")

            else:
                try:
                    with st.spinner("Searching PDF and generating answer..."):
                        # Step 2: Rewrite the question before searching FAISS
                        search_query = rewrite_followup_question(query, api_key)
                        st.caption(f"Search query: {search_query}")

                        results = retrieve_chunks(
                            search_query,
                            index,
                            all_chunks,
                            embedding_model,
                            top_k=3
                        )

                        if not results:
                            st.warning(
                                "No relevant information found. "
                                "Try rephrasing your question."
                            )
                        else:
                            answer = generate_answer(
                                query,
                                results,
                                api_key
                            )

                            #generate question and display the answer
                            #display user question
                            with st.chat_message("user"):
                                st.write(query)

                            #display chatbot answer
                            with st.chat_message("assistant", avatar="📚"):
                                st.write(answer)

                            #display source citations
                            st.write("📚 Retrieved Sources")

                            source_pages = sorted(set(
                                (item["filename"], item["page"])
                                for item in results
                            ))

                            for filename, page in source_pages:
                                st.write(f"📄 Source: {filename} - Page {page}")

                            #display retrieved text chunks
                            st.subheader("🔍 Retrieved Text Chunks")
                            for i, item in enumerate(results, start=1):
                                with st.expander(
                                    f"Chunk {i} - {item['filename']} - Page {item['page']}"
                                ):
                                    st.write(item["text"])

                            #debugging information: display the retrieved context
                            st.write("Retrieved context:", results)

                            #save the question and answer to chat history
                            st.session_state.chat_history.append({
                                "question": query,
                                "answer": answer,
                                "sources": source_pages
                            })

                except Exception as e:
                    st.error(f"Answer generation failed: {e}")

        #step13:display chat history
        st.subheader("💬 Chat History")

        if st.session_state.chat_history:
            for i, chat in enumerate(
                reversed(st.session_state.chat_history), start=1
            ):
                with st.chat_message("user"):
                    st.write(chat["question"])

                with st.chat_message("assistant", avatar="📚"):
                    st.write(chat["answer"])

                    #display source citations in chat history
                    if chat.get("sources"):
                        st.caption("📚 Sources")

                        for filename, page_num in chat["sources"]:
                            st.write(f"📄 Source: {filename} - Page {page_num}")

        else:
            st.info("No chat history yet. Ask your first question!")

        #step14:download chat history
        if st.session_state.chat_history:
            chat_text = ""

            for i, chat in enumerate(
                st.session_state.chat_history, start=1
            ):
                chat_text += f"Question {i}: {chat['question']}\n"
                chat_text += f"Answer {i}: {chat['answer']}\n"

                if chat.get("sources"):
                    chat_text += (
                        "Sources: "
                        + ", ".join(
                            f"{filename} - Page {page_num}"
                            for filename, page_num in chat["sources"]
                        )
                        + "\n"
                    )

                chat_text += "-" * 50 + "\n"

            st.download_button(
                label="📥 Download Chat History",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )

        #step15:clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.warning("No text chunks available. Upload PDFs with extractable text.")

