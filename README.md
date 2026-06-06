# 🛡️ ClaimVision: Intelligent Auto Damage Assessment

ClaimVision is an enterprise-grade AI pipeline that automates vehicle damage assessment and insurance claim processing. By combining advanced Computer Vision (Instance Segmentation) with a Retrieval-Augmented Generation (RAG) agent, it detects car damage from images and cross-references it against insurance policy documents to determine coverage and deductibles.

## 🌟 Key Features
- **Computer Vision Pipeline**: Uses Mask R-CNN with Deformable Convolutional Networks (DCN) to accurately detect and segment 6 types of vehicle damage (dent, scratch, crack, glass shatter, lamp broken, tire flat).
- **RAG-powered Policy Evaluation**: Utilizes `ChromaDB` for semantic search across insurance policies (Master Document, Glass Endorsements, Tire Protection) to map damages to specific clauses.
- **Agentic Workflow**: Powered by `LangGraph` and `GPT-4o`, the agent assesses severity, checks coverages, calculates deductibles, and generates a comprehensive claim report.
- **Premium UI**: A sleek, glassmorphic Streamlit interface displaying original vs. annotated images, clear coverage breakdowns, and policy excerpts.

## 🎥 Demonstration Videos

- **Tire Damage Endorsement Assessment**
https://github.com/user-attachments/assets/b851c058-c3fc-4638-bc92-38854ce1e2b1

- **Complex Damage Assessment (Collision + Glass Coverage)**
https://github.com/user-attachments/assets/fc683577-4a42-45d9-b503-a29a7afb1f70

## 🛠️ Technology Stack
- **Vision**: `MMDetection`, `PyTorch`
- **Agent/RAG**: `LangChain`, `LangGraph`, `OpenAI (GPT-4o)`
- **Vector Store**: `ChromaDB`
- **Frontend**: `Streamlit`
- **Package Management**: `uv`

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `uv` installed (Python package manager)
- OpenAI API Key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/alexiou-alexandros/vision-rag-claims.git
   cd vision-rag-claims
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Set up environment variables:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=sk-your-api-key
   ```

### Running the Application
```bash
uv run streamlit run app.py
```

## 🧠 How It Works
1. **Upload**: User uploads a vehicle image.
2. **Detect Damage**: The MMDetection model runs instance segmentation to identify damages and their confidence scores.
3. **Assess Severity**: Bounding boxes and mask areas are analyzed to categorize the damage as `minor`, `moderate`, or `severe`.
4. **Retrieve Policy**: Semantic search is performed in ChromaDB using natural language queries mapped from the detected damage classes.
5. **Check Coverage**: GPT-4o analyzes the policy chunks to determine if the specific damage is covered and extracts the applicable deductible.
6. **Generate Report**: A final, structured claim report is compiled and presented to the user.

## 🙏 Acknowledgements & Credits
The vision model and test images used in this project are based on the **Car Damage Dataset (CARDD)**. 
- **Dataset**: [Kaggle CARDD Dataset](https://www.kaggle.com/datasets/issamjebnouni/cardd) | [Project Page](https://cardd-ustc.github.io/)
- **Model Weights & Config**: The Mask R-CNN (DCN) config (`dcn_plus_cfg.py`) and pre-trained weights (`best.pth`) were obtained from [Issam Jebnouni's Kaggle Notebook](https://www.kaggle.com/code/issamjebnouni/car-damage-segmentation/output).

## 📄 License
[MIT License](LICENSE)
