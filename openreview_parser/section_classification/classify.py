"""All utility functions for section classification."""
import nltk
import pytorch_lightning as pl
import torch
import torch.nn as nn


from adapters import AutoAdapterModel
from torch.optim import Adam
from transformers import AutoTokenizer

from openreview_parser.utils.data import Paper

SECTIONS = [
    "introduction",
    "background",
    "methodology",
    "experiments and results",
    "conclusion",
]

LABEL2SECTION = {i: section for i, section in enumerate(SECTIONS)}

SECTION_SYNONYMS = {
    "introduction": ["introduction"],
    "background": ["background", "related work", "historical review"],
    "methodology": ["methodology", "method", "algorithm", "properties"],
    "experiments and results": [
        "experiments",
        "results",
        "experiments and results",
        "experimental design",
        "empirical evaluation",
        "experiments and analysis",
        "ablation studies",
        "evaluation",
    ],
    "conclusion": [
        "conclusion",
        "conclusion & discussion",
        "discussion and conclusions",
        "conclusion and outlook",
        "further work",
        "discussions and future directions",
    ],
}


class SectionClassifier(pl.LightningModule):
    """
    SectionClassifier is a PyTorch Lightning module for section classification.

    It uses a transformer-based model to classify sections into different classes.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.

    Attributes:
        num_classes (int): Number of classes for section classification.
        model (SectionClassifierTransformer): Transformer-based model for section classification.
        lr (float): Learning rate for the optimizer.
        loss (nn.CrossEntropyLoss): Loss function for training.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the SectionClassifier class.

        Args:
            config (dict): A dictionary containing the configuration parameters.

        Returns:
            None
        """
        super().__init__()
        self.num_classes = config["model"]["num_classes"]
        self.model = SectionClassifierTransformer(config["model"])
        self.lr = config["lr"]
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def training_step(self, data: dict, data_idx: int) -> torch.Tensor:
        """
        Compute training step for the section classifier.

        Args:
            data (dict): Input data containing features and labels.
            data_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the training step.
        """
        predicted_labels = self.model(data)
        loss = self.loss(predicted_labels, data["labels"])
        acc = (predicted_labels.argmax(dim=1) == data["labels"]).float().mean()
        self.log("training/accuracy", acc)
        return loss

    def validation_step(self, data: dict, data_idx: int) -> torch.Tensor:
        """
        Compute validation step for the section classifier.

        Args:
            data (dict): Input data containing features and labels.
            data_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the validation step.
        """
        predicted_labels = self.model(data)
        loss = self.loss(predicted_labels, data["labels"])
        self.compute_metrics(predicted_labels, data["labels"])
        return loss

    def test_step(self, data: dict, data_idx: int) -> torch.Tensor:
        """
        Compute test step for the section classifier.

        Args:
            data (dict): Input data containing features and labels.
            data_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the test step.
        """
        predicted_labels = self.model(data)
        loss = self.loss(predicted_labels, data["labels"])
        self.compute_metrics(predicted_labels, data["labels"], "test")
        return loss

    def compute_metrics(
        self, predicted_labels: torch.Tensor, labels: torch.Tensor, split="validation"
    ) -> None:
        """
        Compute metrics for the section classifier.

        Args:
            predicted_labels (torch.Tensor): Predicted labels from the model.
            labels (torch.Tensor): True labels.
            split (str, optional): Split name for logging. Defaults to "validation".
        """
        # Compute overall accuracy
        acc = (predicted_labels.argmax(dim=1) == labels).float().mean()
        self.log(f"{split}/accuracy", acc, batch_size=predicted_labels.shape[0])

        # Compute accuracy per class
        for cl in range(self.num_classes):
            mask = labels == cl
            if mask.sum() > 0:
                class_acc = (predicted_labels[mask].argmax(dim=1) == cl).float().mean()
                self.log(
                    f"{split}/accuracy_{cl}",
                    class_acc,
                    batch_size=int(mask.sum().item()),
                )

    def predict(self, data: dict) -> torch.Tensor:
        """
        Make predictions using the section classifier.

        Args:
            data (dict): Input data containing features.

        Returns:
            torch.Tensor: Predicted labels.
        """
        predicted_labels = self.model(data)
        return predicted_labels

    def load_preprocessing_utils(self, device: str) -> None:
        """
        Load preprocessing utilities for the section classifier.

        Args:
            device (str): Device to load the utilities on.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.embedding_model = AutoAdapterModel.from_pretrained(
            "allenai/specter2_base"
        ).to(device)
        self.embedding_model.load_adapter(
            "allenai/specter2", source="hf", load_as="classification", set_active=True
        )
        self.embedding_model = self.embedding_model.to(device)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the section classifier.

        Returns:
            torch.optim.Optimizer: Optimizer for training.
        """
        return Adam(self.model.parameters(), lr=self.lr)


class SectionClassifierTransformer(nn.Module):
    """
    A transformer-based section classifier.

    Args:
        config (dict): Configuration parameters for the classifier.

    Attributes:
        projection (nn.Linear): Linear layer for projecting input embeddings.
        transformer (nn.TransformerEncoder): Transformer encoder for sequence modeling.
        label_predictor (nn.Linear): Linear layer for predicting section labels.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the Classify model.

        Args:
            config (dict): A dictionary containing the configuration parameters.

        Returns:
            None
        """
        super().__init__()
        self.projection = nn.Linear(768, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            dropout=config["dropout"],
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )
        self.label_predictor = nn.Linear(512, config["num_classes"])

    def forward(self, data: dict) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            data (dict): A dictionary containing the input data.
                - embeddings (torch.Tensor): Input embeddings.
                - mask (torch.Tensor): Padding mask for the embeddings.

        Returns:
            torch.Tensor: Predicted labels for the input data.
        """
        embeddings = self.projection(data["embeddings"])
        transformed_embeddings = self.transformer(
            embeddings, src_key_padding_mask=data["mask"]
        )
        return self.label_predictor(transformed_embeddings.mean(dim=1))


def classify_sections(
    paper: Paper, section_classifier: SectionClassifier, config: dict
) -> Paper:
    """
    Classify the sections of a given paper based on their content.

    Args:
        paper (Paper): The paper object containing the structured content.
        section_classifier (SectionClassifier): The section classifier model.
        config (dict): Configuration parameters for the classification process.

    Returns:
        Paper: The paper object with the classified sections.
    """
    for key, section in paper.structured_content.items():
        section_classified = False
        # Check whether section is in synonyms
        for key, synonyms in SECTION_SYNONYMS.items():
            if section.name in synonyms:
                section.classification = key
                section_classified = True
                break

        # Otherwise use classifier
        if not section_classified:
            sentences = nltk.tokenize.sent_tokenize(section.text)
            sentences = sentences[: config["max_n_sentences_per_example"]]
            model_inputs = section_classifier.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(config["device"])

            with torch.no_grad():
                outputs = section_classifier.embedding_model(**model_inputs)
                model_output = outputs.last_hidden_state.mean(dim=1)

            section_classifier_input = {
                "embeddings": model_output.unsqueeze(0),
                "mask": torch.zeros(1, model_output.shape[0]).to(config["device"]),
                "label": 0,
            }
            predicted_labels = section_classifier.predict(section_classifier_input)
            predicted_label = int(predicted_labels.argmax(dim=1).cpu().item())
            section.classification = LABEL2SECTION[predicted_label]

    return paper
