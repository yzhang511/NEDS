# This file contains the implementation of the r2 score metric
from torcheval.metrics import R2Score
import torch
import torch.nn.functional as F

r2_metric = R2Score()
def r2_score(y_true, y_pred, device="cpu"):
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def top_k_accuracy(logits, targets, k=1):
    """
    Computes the top-k accuracy for the given logits and targets.
    
    :param logits: Tensor of logits from the model (shape [batch_size, num_classes])
    :param targets: Ground truth labels (shape [batch_size])
    :param k: Top k predictions to consider for accuracy
    :return: Top-k accuracy as a Python float
    """
    # Get the top k predictions from logits; indices is [batch_size, k]
    _, indices = torch.topk(logits, k, dim=1)

    predicted_one_hot = torch.zeros_like(logits).scatter_(1, indices, 1)
    # Calculate correct predictions by element-wise multiplication of predicted_one_hot and targets
    correct = (predicted_one_hot * targets).sum(dim=1)
    # Check if any of the top-k predictions was correct (sum > 0)
    correct = correct > 0
    # Calculate the mean accuracy over the batch
    top_k_acc = correct.float().mean().item()
    return top_k_acc

def clip_contrastive_loss(similarity_matrix):
    """
    Compute CLIP's contrastive loss given a similarity matrix.
    The matrix contains cosine similarities of two sets of features.
    """
    labels = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
    percent_correct = topk(similarity_matrix, labels, k=1)
    loss_i = F.cross_entropy(similarity_matrix, labels)
    loss_t = F.cross_entropy(similarity_matrix.t(), labels)
    return (loss_i + loss_t) / 2, percent_correct