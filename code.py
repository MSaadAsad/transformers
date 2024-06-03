import random
import torch
import tqdm
import numpy as np
from scipy.stats import spearmanr

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from utils import get_mnist_dataloader, construct_mlp, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader_fn = get_mnist_dataloader
train_loader = dataloader_fn(
    batch_size=512, split="train", shuffle=False, subsample=True
)

def train(
    model,
    loader,
    lr=0.1,
    epochs=10,
    momentum=0.9,
    weight_decay=1e-4,
    label_smoothing=0.0,
    save_name="default",
    model_id=0,
    save=True,
):
    if save:
        torch.save(model.state_dict(), f"LDS_checkpoints/{save_name}_{model_id}_epoch_0.pt")
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(epochs):
        # We use consistent data ordering when training.
        set_seed(model_id * 10_061 + epoch + 1)
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
        if save:
            torch.save(
                model.state_dict(),
                f"checkpoints/{save_name}_{model_id}_epoch_{epoch}.pt",
            )
    return model

class LDS():
    def __init__(self, model, dataloader, scorer=None, scores_path=None):
        self.model = model
        self.scorer = scorer
        self.dataloader = dataloader
        self.scores = self.load_scores(scores_path) if scores_path else None

    def load_scores(self, scores_path):
        return torch.load(scores_path)

    def generate_subset(self, alpha):
        subset_size = int(alpha * len(self.dataloader.dataset))
        indices = np.random.choice(range(len(self.dataloader.dataset)), size=subset_size, replace=False)
        subset_loader = torch.utils.data.DataLoader(
            self.dataloader.dataset, 
            batch_size=self.dataloader.batch_size, 
            sampler=torch.utils.data.SubsetRandomSampler(indices)
        )
        return subset_loader, indices

    def gen_score(self, data):
        if self.scores is None:
            self.scores = self.scorer(data)
        return self.scores

    def train_models(self, alpha, num_models=5, lr=0.1, epochs=10, momentum=0.9, weight_decay=1e-4, save_name="model"):
        models = []
        for i in range(num_models):
            subset_loader, _ = self.generate_subset(alpha)
            model = self.model(seed=i).to(DEVICE)
            trained_model = train(
                model,
                subset_loader,
                lr=lr,
                epochs=epochs,
                momentum=momentum,
                weight_decay=weight_decay,
                save_name=str(i),
                model_id=i,
                save=True
            )
            models.append(trained_model)
        return models

    def compute_lds(self, alpha, num_subsets=5):
        total_score = 0

        for _ in range(num_subsets):
            subset_loader, indices = self.generate_subset(alpha)
            subset_scores = self.gen_score(subset_loader) if self.scores is None else [self.scores[i] for i in indices]
            subset_outputs = [model(subset_loader.dataset[i][0].unsqueeze(0)) for i in indices]

            subset_outputs = torch.cat([output.view(-1) for output in subset_outputs]).cpu().numpy()
            subset_scores = np.array(subset_scores)

            total_score += spearmanr(subset_scores, subset_outputs).correlation

        return total_score / num_subsets

train_loader = get_mnist_dataloader(
    batch_size=512, split="train", shuffle=False, subsample=True
)

scorer = LDS(construct_mlp, dataloader=train_loader, scores_path="if_logix.pt")
models = scorer.train_models(alpha=0.1)

#scores_path = "if_logix.pt"

#scorer = LDS(construct_mlp, dataloader = train_loader, scores_path = scores_path)
#performance = scorer.compute_lds(0.1)
#print("Score: ", performance)

"""

"""
