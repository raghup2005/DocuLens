# DocuLens 🔍

**Chat with your documents, including their images, charts and tables.**

DocuLens is a multimodal Retrieval-Augmented Generation (RAG) application. Normal RAG only understands text. DocuLens also reads the images and charts inside your documents, so you can ask questions whose answers live in a picture, not just a paragraph. Answers are grounded in your own files and shown with their sources.

---

## Features

- **Multimodal ingestion**: reads text and images from your documents
- **Image understanding**: a vision-capable Gemini model turns each image or chart into a searchable text description
- **Semantic search**: finds the pieces closest in meaning to your question, not just keyword matches
- **Grounded answers**: the model answers only from retrieved content and says "I don't know" when the answer isn't there
- **Source display**: every answer shows which documents (and images) it came from
- **Simple web UI**: chat interface built with Gradio

---

## How It Works

```
                 ┌───────────────────────── INGESTION (run once) ─────────────────────────┐
 Documents  ───► │ Load files → Split into text + images → Chunk text                     │
 (text, images)  │ Images → Gemini vision → text descriptions                              │
                 │ All pieces → Embeddings → Store in ChromaDB (with metadata)            │
                 └────────────────────────────────────────────────────────────────────────┘

                 ┌───────────────────────── QUERY (every question) ───────────────────────┐
 User question ─►│ Embed question → Top-k similarity search in ChromaDB                   │
                 │ Retrieved text + image info → Prompt → Gemini → Answer + sources       │
                 └────────────────────────────────────────────────────────────────────────┘
```

**Step by step**

1. **Load**: documents are read from the data folder.
2. **Split**: content is separated into text and images.
3. **Chunk**: text is cut into small overlapping pieces so context isn't lost at the edges.
4. **Describe images**: each image is sent to Gemini, which writes a text description of it (including charts, labels and trends).
5. **Embed**: every text chunk and image description is converted into an embedding, a list of numbers that captures meaning.
6. **Store**: embeddings go into ChromaDB along with metadata (source file, type, page or image path).
7. **Retrieve**: your question is embedded with the *same* embedding model, and the closest pieces are returned.
8. **Generate**: the retrieved context is placed in a prompt, and Gemini writes an answer using only that context.
9. **Show**: the answer appears in the Gradio app with its sources.

---

## Tech Stack

| Part | Tool |
|---|---|
| Language | Python |
| Orchestration | LangChain |
| Vector database | ChromaDB |
| Embeddings + LLM + vision | Google Gemini API |
| User interface | Gradio |

---

## Project Structure

```
DocuLens/
├── ingestion_pipeline.py    # loads docs, processes images, creates embeddings, fills ChromaDB
├── retrieval_pipeline.py    # embeds the question and fetches the most relevant pieces
├── chat_pipeline.py         # builds the prompt and generates the grounded answer
├── app_gradio.py            # Gradio web interface
├── docs/                    # your input documents (text and images)
├── db/                      # ChromaDB storage (created automatically)
├── requirements.txt
├── .env                     # your API key (never commit this)
└── README.md
```

---

## Setup

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd DocuLens
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key**

Get a free key from Google AI Studio, then create a `.env` file in the project folder:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## Usage

**Step 1: Add your documents** to the `docs/` folder.

**Step 2: Build the knowledge base** (run once, and again whenever you add new documents):

```bash
python ingestion_pipeline.py
```

**Step 3: Start the app:**

```bash
python app_gradio.py
```

Open the local link Gradio prints in the terminal (usually `http://127.0.0.1:7860`) and start asking questions.

---

## Example Questions

- "What does the revenue chart show about the last four quarters?"
- "Summarize the main points of the architecture diagram."
- "Which document mentions the product launch, and what does it say?"

---

## Key Design Decisions

- **Image descriptions instead of raw image embeddings.** Converting images to text lets the project reuse a standard text-based RAG stack and makes retrieval easy to debug. The tradeoff is that a description can miss fine visual detail.
- **Same embedding model for documents and questions.** Different models create different vector spaces, so mixing them would make similarity scores meaningless.
- **Strict prompting.** The model is told to answer only from the retrieved context, which reduces hallucination.
- **Metadata on every chunk.** Source file, content type and location are stored so answers can be cited and checked.
- **Gradio over Streamlit.** Streamlit rendered a blank page in the development environment, so the UI was switched to Gradio, which worked reliably.

---

## Limitations

- Image quality depends on the vision model's description; small text or dense charts may lose detail.
- Complex or nested tables may not be parsed perfectly.
- Free API tiers have rate limits, so large ingestion runs can be slow.
- No reranking or hybrid (keyword + vector) search yet.

---

## Future Improvements

- Hybrid search (keyword + vector) for exact-term matches
- A reranker to improve precision of the top results
- Layout-aware parsing for tables and multi-column pages
- Page-level image retrieval (ColPali-style) for visually rich documents
- Conversation memory for follow-up questions
- An evaluation suite measuring retrieval accuracy and answer faithfulness
- Support for more file types (audio, video transcripts)

---

## License

MIT License. Feel free to use and modify.

---

## Author

**Raghu**
B.Tech Computer Science Engineering, Jain University, Bangalore