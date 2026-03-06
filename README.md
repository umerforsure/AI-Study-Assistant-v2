AI Study Assistant v2
Hybrid Retrieval-Augmented Question Answering System with Fine-Tuned T5 and Qwen Reasoning Model
________________________________________
Project Overview
AI Study Assistant v2 is an intelligent academic assistant designed to help students interact with their study materials. The system allows users to upload documents and ask questions directly from the content, generate summaries, and create structured study notes.
The project implements a hybrid architecture combining a custom fine-tuned T5 transformer model with the Qwen 2.5 Instruct model. Retrieval-Augmented Generation techniques are used to ensure answers remain grounded in the uploaded document.
The system processes lecture slides, research papers, and academic notes to provide concise and context-aware responses.
________________________________________
Key Features
•	Document-based question answering
•	Retrieval-Augmented Generation pipeline
•	Fine-tuned transformer model for academic QA
•	Hybrid reasoning with fallback model
•	Automatic summarization of study material
•	Bullet-point study note generation
•	Multi-format document support
•	Local GUI interface for easy interaction
Supported file types:
•	PDF
•	DOCX
•	PPTX
•	TXT
•	Images (via OCR)
________________________________________
System Architecture
The system follows a hybrid pipeline where document embeddings are used for retrieval while transformer models handle response generation.
Pipeline steps:
1.	User uploads a document.
2.	Text is extracted and divided into chunks.
3.	Sentence-BERT generates embeddings for each chunk.
4.	FAISS performs similarity search to retrieve relevant contexts.
5.	The fine-tuned T5 model generates the initial answer or summary.
6.	If the output is weak or incomplete, Qwen 2.5 refines the response.
7.	The final structured output is returned to the user.
This approach ensures responses remain grounded in the document while benefiting from strong reasoning capabilities.
________________________________________
Interface Preview
The application includes a local graphical interface where users can upload documents, enable retrieval mode, and interact with the AI assistant.
Interface tools include:
•	RAG toggle for document-grounded responses
•	Document upload
•	Summarization
•	Study notes generation
•	Chat-style question answering
________________________________________
Model Training
The T5 model was fine-tuned using publicly available question-answering datasets.
Datasets used:
•	SQuAD v2.0
•	SciQ dataset
Training preparation included:
•	Instruction style formatting
•	Tokenization using the T5 tokenizer
•	Label masking for padded tokens
•	Train and validation split
The goal of training was to specialize the model for short academic question answering and concise explanations.
________________________________________
Evaluation
Because the system generates natural language responses, traditional classification metrics such as accuracy or confusion matrices are not suitable.
Instead, the following evaluation metrics were used:
Metric	Score
ROUGE-1	0.464
ROUGE-2	0.428
ROUGE-L	0.463
Average Semantic Similarity	0.54
ROUGE measures lexical overlap between generated and reference answers, while semantic similarity evaluates meaning-based alignment.
________________________________________
Technologies Used
Core libraries and frameworks used in this project:
•	Python
•	PyTorch
•	HuggingFace Transformers
•	Sentence Transformers
•	FAISS
•	Tesseract OCR
•	LangChain style RAG pipeline
•	T5 Transformer
•	Qwen 2.5 Instruct Model
________________________________________
Installation
Clone the repository:
git clone https://github.com/yourusername/AI-Study-Assistant-v2.git
cd AI-Study-Assistant-v2
Install dependencies:
pip install -r requirements.txt
________________________________________
Model Setup
Due to size limitations, pretrained models are not included in the repository.
Download the required models manually:
T5 Fine-Tuned Model
Qwen2.5 1.5B Instruct
Place them inside the following directories:
models/
│
├── t5_finetuned
└── qwen
Update the model paths inside the code if needed.
________________________________________
Running the Project
Run the backend script:
python ai_study_assistant2.py
The interface will launch locally, allowing document upload and interaction with the AI assistant.
________________________________________
Strengths of the System
•	Hybrid model architecture improves answer quality
•	Retrieval pipeline ensures grounded responses
•	Supports multiple document formats
•	Lightweight custom model reduces computation
•	Local execution without cloud dependency
________________________________________
Limitations
•	Qwen inference can be slow on CPU systems
•	Long documents require chunking
•	Performance depends on the clarity of uploaded documents
________________________________________
Future Improvements
Potential future enhancements include:
•	GPU acceleration support
•	Web-based deployment
•	Multi-document retrieval
•	Vector database integration
•	Improved UI and conversation memory
________________________________________
Author
Muhammad Umer Ijaz
BS Artificial Intelligence
Government College University Lahore
________________________________________
License
This project is intended for academic and educational use.

