# 📝 OpenReview Parser

![Mypy](https://img.shields.io/badge/mypy-checked-blue)
![pydocstyle](https://img.shields.io/badge/pydocstyle-passing-brightgreen)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains a pipeline that parses the submissions and reviews of all OpenReview venues into a unified format and annotates them with metadata.

🔍 Metadata includes:
* 🧪 **Research Hypothesis** (annotated via LLM)
* 🔗 **References** (from [Semantic Scholar](https://www.semanticscholar.org/))
* 📊 **Citation Counts** for accepted papers (from [Semantic Scholar](https://www.semanticscholar.org/))

---

## 📦 Dependencies 

Create a new conda environment and install the dependencies:

```bash
conda create -n openreview_parser python=3.11
conda activate openreview_parser
pip install -e . 
python -c "import nltk; nltk.download('punkt_tab')"
```

---

## 📄 GROBID Setup 

We parse PDF submissions using [GROBID](https://github.com/kermitt2/grobid).

⚙️ From the `./openreview_parser/pipeline` directory:

```bash
bash ./scripts/setup_grobid.sh
```

▶️ To test the setup:

```bash
bash ./scripts/run_grobid.sh
```

💡 **Tip**: GROBID requires Java (≥11). Check with `java -version`.

🐛 If you run into trouble, check issues in the [s2orc-doc2json repo](https://github.com/allenai/s2orc-doc2json).

🪟 **tmux** is also needed:

```bash
sudo apt update
sudo apt install tmux
```

---

## 📚 Data Model

📖 Overview of the unified data model: [README](./openreview_parser/utils/README.md)

---

## 🚀 Run Pipeline 

If you do not have an OpenReview account, run the pipeline with the **guest client** by setting the corresponding field in the config files in `./openreview_parser/pipeline/configs/`.

🛠️ Pipeline steps:
1. 📥 Retrieve venues from OpenReview
2. 🧩 Map venue schemas to a unified data model
3. 📄 Parse all submissions and reviews
4. 🏷️ Annotate submissions with metadata:
   - 🔗 References (Title & Abstract)
   - 🧪 Research Hypothesis
   - 📊 Citation Counts

---

### 🏃 Run Script

From the `./openreview_parser/pipeline` directory:

```bash
bash ./scripts/run_pipeline.sh
```

💡 _This will run the pipeline twice: once for API V1 venues and once for API V2._

⚠️ _Metadata annotation is **optional** and controlled in [`pipeline.yaml`](./openreview_parser/pipeline/configs/pipeline.yaml)._

---

## 🧠 Metadata Annotation

* 🧪 **Hypothesis Annotation**: Set `OPENAI_API_KEY` in your environment.
* 📊 **Citation Count**: Recommended to use a Semantic Scholar API key to avoid rate limits.
* 🔗 **References**:
    Run from `./openreview_parser/scientific_databases`:

    ```bash
    python s2_datasets.py
    ```

    💾 Requires ~140GB disk space and a Semantic Scholar API key.

---

## 🧪 Run for Specific Venues

Make sure GROBID is running. Then:

**API V1 venue**:
```bash
python pipeline.py --config ./configs/pipeline_v1.yaml --venue "ICLR.cc/2022/Conference"
```

**API V2 venue**:
```bash
python pipeline.py --config ./configs/pipeline_v2.yaml --venue "ICLR.cc/2024/Conference"
```

📄 A list of API V1 venues: [`venue_strings_api_v1.json`](./openreview_parser/pipeline/data/venue_strings_api_v1.json)

---

## 📊 Dataset 

You can find the resulting dataset on HuggingFace:  
👉 [scientific-quality-score-prediction](https://huggingface.co/datasets/nhop/scientific-quality-score-prediction)

---

## 🕒 Updates 

🗓️ Pipeline run timestamps:
* ✅ Latest: **1.1.2025**

---

## 📚 Citation 

If you use this codebase, please cite:

```bibtex
@article{hopner2025automatic,
  title={Automatic Evaluation Metrics for Artificially Generated Scientific Research},
  author={H{"o}pner, Niklas and Eshuijs, Leon and Alivanistos, Dimitrios and Zamprogno, Giacomo and Tiddi, Ilaria},
  journal={arXiv preprint arXiv:2503.05712},
  year={2025}
}
```
