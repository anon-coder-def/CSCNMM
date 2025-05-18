import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mimic4noisy.gaussian_model import *
from utils import *
from baseline.LossFunction.Loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
def standardization_minmax(data):
    mini = np.min(data, axis=0)
    maxi = np.max(data, axis=0)
    return (data - mini) / (maxi-mini)



def normalize(data):
    """
    Normalize data to range [0, 1] for both numpy arrays and torch tensors.
    
    Args:
        data (np.ndarray or torch.Tensor): Input data.
    Returns:
        np.ndarray or torch.Tensor: Normalized data.
    """
    if isinstance(data, np.ndarray):
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
    elif isinstance(data, torch.Tensor):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + 1e-9)
    else:
        raise TypeError("Unsupported data type. Use numpy.ndarray or torch.Tensor.")



def select_class_by_class(model_loss,loss_all,pred_all,args,epoch,x_idxs,labels, task_index):
    '''
        single class confident sample selection EPS
    '''
    gamma = 0.5

    if args.mean_loss_len > epoch:
        loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
            loss_all[x_idxs, :epoch].mean(axis=1))
    else:
        if args.mean_loss_len < 2:
            loss_mean = loss_all[x_idxs, epoch]
        else:
            loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
                loss_all[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))

    # STANDARDIZE LOSS FOR EACH CLASS
    labels_numpy = labels.detach().cpu().numpy().squeeze()
    recreate_idx=torch.tensor([]).long()
    batch_idxs = torch.tensor(np.arange(len(model_loss))).long()
    standar_loss = np.array([])
    
    
    for i in range(args.nbins[torch.unique(task_index)]):
        if (labels_numpy==i).sum()>1:
            if args.standardization_choice == 'z-score':
                each_label_loss = standardization(loss_mean[labels_numpy==i])
            else:
                each_label_loss = standardization_minmax(loss_mean[labels_numpy == i])
                
                
            standar_loss = np.concatenate((standar_loss,each_label_loss))
            recreate_idx=torch.cat((recreate_idx,batch_idxs[labels_numpy==i]))
        elif (labels_numpy==i).sum()==1:
            standar_loss = np.concatenate((standar_loss, [0.]))
            recreate_idx=torch.cat((recreate_idx,batch_idxs[labels_numpy==i]))

    # SELECT CONFIDENT SAMPLES
    
    _, model_sm_idx = torch.topk(torch.from_numpy(standar_loss), k=int(standar_loss.size*(standar_loss<=standar_loss.mean()).mean()), largest=False)

    model_sm_idxs = recreate_idx[model_sm_idx]
    

    # SELECT LESS CONFIDENT SAMPLES 
    _, less_confident_idx = torch.topk(torch.from_numpy(standar_loss), k=int(standar_loss.size * (standar_loss > standar_loss.mean()).mean()), largest=True)
    less_confident_idxs = recreate_idx[less_confident_idx]

    # CALCULATING L_CONF
    model_loss_filter = torch.zeros((model_loss.size(0))).to(device)
    model_loss_filter[model_sm_idxs] = 1.0
    L_conf = (model_loss_filter * model_loss).mean()

    return L_conf, model_sm_idxs, less_confident_idxs




from sklearn.mixture import GaussianMixture
def select_multi_class_inst(model_loss,loss_all,pred_all,rank_all,args,epoch,x_idxs,labels, task_index):
    '''
        Based on average of rank and loss
    '''
    gamma = 0.5
    alpha = 0.5

    # If current epoch is less than mean_loss_len, use all available epochs
    if args.mean_loss_len > epoch + 1:
        # Use all available historical epochs
        historical_mean = loss_all[x_idxs, :, :epoch + 1].mean(axis=2)  # Shape: (128, 25)
        historical_rank = rank_all[x_idxs, :, :epoch + 1].mean(axis=2)  # Shape: (128, 25)
        loss_mean = gamma * loss_all[x_idxs, :, epoch] + (1 - gamma) * historical_mean  # Shape: (128, 25)
        rank_mean = gamma * rank_all[x_idxs, :, epoch] + (1 - gamma) * historical_rank  # Shape: (128, 25)
    else:
        # Use only the last `mean_loss_len` epochs
        historical_mean = loss_all[x_idxs, :, (epoch - args.mean_loss_len + 1):epoch + 1].mean(axis=2)  # Shape: (128, 25)
        historical_rank = rank_all[x_idxs, :, (epoch - args.mean_loss_len + 1):epoch + 1].mean(axis=2)
        loss_mean = gamma * loss_all[x_idxs, :, epoch] + (1 - gamma) * historical_mean  # Shape: (128, 25)
        rank_mean = gamma * rank_all[x_idxs, :, epoch] + (1 - gamma) * historical_rank

    # Combine two metrics
    loss_rank_mean = alpha * loss_mean + (1-alpha)*rank_mean


    # Stage 1:  Select top k indices for each label (1-25) for both 1 and 0 class    
    labels_numpy = labels.detach().cpu().numpy().squeeze()
    # Container for top-k results
    top_k_indices_appear = {}
    top_k_indices_disappear = {}
    k = 10

    # Process each label
    num_labels = labels_numpy.shape[1]
    for i in range(num_labels):  # Iterate over each label (25 in this case)
        # Separate appear (class 1) and disappear (class 0) indices
        appear_idx = (labels_numpy[:, i] == 1)
        disappear_idx = (labels_numpy[:, i] == 0)

        # Standardize losses for appear instances
        if appear_idx.sum() > 0:  # Ensure there are "appear" instances
            appear_losses = loss_rank_mean[appear_idx, i]
            if args.standardization_choice == 'z-score':
                standardized_appear_losses = standardization(appear_losses)
            else:
                standardized_appear_losses = standardization_minmax(appear_losses)
            
            # Get top-k indices from the standardized losses
            top_k_idx_appear = appear_idx.nonzero()[0][np.argsort(standardized_appear_losses)[:k]]
            top_k_indices_appear[i] = top_k_idx_appear.tolist()

        # Standardize losses for disappear instances
        if disappear_idx.sum() > 0:  # Ensure there are "disappear" instances
            disappear_losses = loss_rank_mean[disappear_idx, i]
            if args.standardization_choice == 'z-score':
                standardized_disappear_losses = standardization(disappear_losses)
            else:
                standardized_disappear_losses = standardization_minmax(disappear_losses)
            
            # Get top-k indices from the standardized losses
            top_k_idx_disappear = disappear_idx.nonzero()[0][np.argsort(standardized_disappear_losses)[:k]]
            top_k_indices_disappear[i] = top_k_idx_disappear.tolist()
              
    
    # CALCULATING L_CONF
    model_loss_filter = torch.zeros(model_loss.shape).to(device)

    # Update model_loss_filter with confident indices
    for class_idx in top_k_indices_appear.keys():  # Iterate over each class (column)
        # Get the top-k row indices for "appear" and "disappear" for this class
        appear_indices = top_k_indices_appear[class_idx]
        disappear_indices = top_k_indices_disappear[class_idx]

        # Combine indices (union of "appear" and "disappear")
        confident_indices = torch.tensor(appear_indices + disappear_indices, device=model_loss_filter.device).unique()

        # Update model_loss_filter for this class
        model_loss_filter[confident_indices, class_idx] = 1.0
        
        
    L_conf = (model_loss_filter * model_loss).mean()
    return L_conf





def calculate_label_similarity(labels):
    """
    Calculate label similarity matrix based on co-occurrence.
    
    Args:
        labels (np.ndarray): Multi-hot label matrix, shape (n_samples, n_labels).
    
    Returns:
        np.ndarray: Label similarity matrix, shape (n_labels, n_labels).
    """
    co_occurrence = np.dot(labels.T, labels)  # Co-occurrence matrix
    frequency = np.sum(labels, axis=0)  # Label frequencies
    similarity = co_occurrence / (np.sqrt(np.outer(frequency, frequency)) + 1e-9)
    return similarity


def calculate_diversity(probabilities, labels, alpha=1.0, beta=1.0, gamma=0.5):
    """
    Calculate selection metric combining uncertainty, rank, and contextual diversity.
    
    Args:
        probabilities (np.ndarray): Predicted probabilities, shape (n_samples, n_labels).
        loss (np.ndarray): Loss values for each instance-label pair, shape (n_samples, n_labels).
        labels (np.ndarray): Ground truth labels, shape (n_samples, n_labels).
        alpha (float): Weight for uncertainty.
        beta (float): Weight for rank.
        gamma (float): Weight for contextual diversity.
    
    Returns:
        np.ndarray: Selection metric scores, shape (n_samples, n_labels).
    """
    # Calculate uncertainty (entropy)
    epsilon = 1e-9
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    # Calculate contextual diversity
    similarity_matrix = calculate_label_similarity(labels)
    diversity = 1 - np.dot(probabilities, similarity_matrix) / (np.sum(probabilities, axis=1, keepdims=True) + epsilon)

    # Combine metrics
    selection_metric = diversity
    return selection_metric




def select_memorization_and_forgetting(pred_all, labels, clean_labels, x_ids, epoch, lambda_coeff=1.0):
    """
    Calculate memorization (M), forgetting (F), and selection metric (C) for multi-label data.
    
    Args:
        pred_all (torch.Tensor): Historical predictions, shape (n_samples, n_bins, n_epochs).
        labels (torch.Tensor): True labels, shape (n_samples, n_bins).
        lambda_coeff (float): Coefficient for weighting forgetting in the selection metric.
    
    Returns:
        memorization (torch.Tensor): Memorization difficulty, shape (n_samples, n_bins).
        forgetting (torch.Tensor): Forgetting difficulty, shape (n_samples, n_bins).
        selection_metric (torch.Tensor): Combined metric, shape (n_samples, n_bins).
    """
    
    # Stage 1:  Select top k indices for each label (1-25) for both 1 and 0 class    
    labels_numpy = labels.detach().cpu().numpy().squeeze() 

    batch_size, n_bins = labels_numpy.shape
    
    # Initialize memorization and forgetting difficulty
    memorization = torch.zeros((batch_size, n_bins), device = device)
    forgetting = torch.zeros((batch_size, n_bins), device = device)
    
    curr_pred_all = pred_all[x_ids, :, :epoch]


    for i in range(batch_size):  # Iterate over each sample
        for j in range(n_bins):  # Iterate over each label (bin)
            # Extract prediction sequence for sample and label
            pred_sequence = [int(round(pred)) for pred in curr_pred_all[i, j, :].tolist()]  # Convert predictions to integers (0 or 1)
            true_label = int(labels[i, j])

            # Convert prediction sequence to binary (1 if correct, 0 otherwise)
            binary_sequence = [1 if pred == true_label else 0 for pred in pred_sequence]


            # Identify segments of memorized (1) and misclassified (0)
            segments = []
            current_segment = [binary_sequence[0]]
            for pred in binary_sequence[1:]:
                if pred == current_segment[-1]:
                    current_segment.append(pred)
                else:
                    segments.append(current_segment)
                    current_segment = [pred]
            segments.append(current_segment)

            # Calculate memorization and forgetting difficulties
            mem_segments = [seg for seg in segments if seg[0] == 0]  # Misclassified segments
            forget_segments = [seg for seg in segments if seg[0] == 1]  # Memorized segments
            
            if len(mem_segments) > 0:
                memorization[i, j] = sum(len(seg) for seg in mem_segments) / len(mem_segments)
            if len(forget_segments) > 0:
                forgetting[i, j] = sum(len(seg) for seg in forget_segments) / len(forget_segments)

    # Calculate selection metric
    selection_metric = memorization - lambda_coeff * forgetting 
    
    
    return memorization, forgetting, torch.tensor(selection_metric, device = device)



def select_memorization_and_forgetting_per_epoch(pred_all, labels, clean_labels, epoch, lambda_coeff=1.0):

    batch_size, n_bins = labels.shape
    
    # Initialize memorization and forgetting difficulty
    memorization = torch.zeros((batch_size, n_bins), device = device)
    forgetting = torch.zeros((batch_size, n_bins), device = device)
    
    curr_pred_all = pred_all[:, :, :epoch]

    for i in range(batch_size):  # Iterate over each sample
        for j in range(n_bins):  # Iterate over each label (bin)
            # Extract prediction sequence for sample and label
            pred_sequence = [int(round(pred)) for pred in curr_pred_all[i, j, :].tolist()]  # Convert predictions to integers (0 or 1)
            true_label = int(labels[i, j])

            # Convert prediction sequence to binary (1 if correct, 0 otherwise)
            binary_sequence = [1 if pred == true_label else 0 for pred in pred_sequence]


            # Identify segments of memorized (1) and misclassified (0)
            segments = []
            current_segment = [binary_sequence[0]]
            for pred in binary_sequence[1:]:
                if pred == current_segment[-1]:
                    current_segment.append(pred)
                else:
                    segments.append(current_segment)
                    current_segment = [pred]
            segments.append(current_segment)

            # Calculate memorization and forgetting difficulties
            mem_segments = [seg for seg in segments if seg[0] == 0]  # Misclassified segments
            forget_segments = [seg for seg in segments if seg[0] == 1]  # Memorized segments
            
            if len(mem_segments) > 0:
                memorization[i, j] = sum(len(seg) for seg in mem_segments) / len(mem_segments)
            if len(forget_segments) > 0:
                forgetting[i, j] = sum(len(seg) for seg in forget_segments) / len(forget_segments)

    # Calculate selection metric
    
    selection_metric = memorization - lambda_coeff * forgetting 
    
    
    
    
    
    return memorization, forgetting, torch.tensor(selection_metric, device = device)
    







def fit_gaussian_model_loss_corr_perclass(args, loss, rank, C, criterion_now_each, labels, pred_labels, true_labels):
    """
    Perform correction for noisy labels using GMM fitting for positive pairs per class and a single GMM for negative pairs.

    Args:
        loss (torch.Tensor): Per-sample loss (shape: [N, D]).
        rank (torch.Tensor): Per-sample rank (shape: [N, D]).
        C (torch.Tensor): Correlation matrix (shape: [D, D]).
        criterion_now_each (function): Loss function to compute corrections.
        labels (torch.Tensor): Binary noisy labels (shape: [N, D]).
        pred_labels (torch.Tensor): Predicted probabilities (shape: [N, D]).
        true_labels (torch.Tensor): Ground truth labels (shape: [N, D]).

    Returns:
        torch.Tensor: Corrected loss.
    """
    # Normalize the selection metric
    pos_selection_metric = normalize(C)
    neg_selection_metric =  normalize(C)


    threshold = args.threshold_coef
    # Control EMA coefficient
    batch_size, num_classes = labels.shape


    # Compute probabilities for negative labels
    Corr = get_correlation_matrix(labels)
    negative_corr = labels @ Corr
    negative_pair_mask = (negative_corr > threshold) & (labels == 0)

    # # Initialize clean mask
    clean_mask = torch.zeros_like(labels, dtype=torch.float32)
    soft_correction_weight = 0.5  # Adjust this weight as needed for soft correction

    # Iterate over each class for positive GMM fitting
    for class_idx in range(num_classes):

        class_pos_indices = labels[:, class_idx] > 0.5
        class_neg_indices = labels[:, class_idx] <= 0.5
        
        # Correct Correlated Negative Labels ONLY
        class_neg_pair_indices = negative_pair_mask[:, class_idx] == 1
        
        class_selection_metric = pos_selection_metric[class_pos_indices, class_idx]
        neg_class_selection_metric = neg_selection_metric[class_neg_indices, class_idx]

        if len(class_selection_metric) > 1:
            # Fit GMM for positive samples in this class
            class_selection_metric_np = class_selection_metric.detach().cpu().numpy()
            pos_gmm = fit_gaussian_model(class_selection_metric_np)
            pos_clean_mean, pos_noisy_mean = np.sort(pos_gmm.means_.flatten())
            print("Positive GMM Means:", pos_gmm.means_.flatten())

            ############################## POSITIVE CORRECTION #############################
            pos_clean_indices = class_pos_indices & (pos_selection_metric[:, class_idx] <= pos_clean_mean)
            pos_uncertain_indices = class_pos_indices & (pos_clean_mean < pos_selection_metric[:, class_idx]) & (pos_selection_metric[:, class_idx] <= pos_noisy_mean)
            pos_noisy_indices = class_pos_indices & (pos_selection_metric[:, class_idx] > pos_noisy_mean)
            
            # Update clean mask and store clean indices
            clean_mask[pos_clean_indices, class_idx] = 1  # Mark positives as clean

            labels[pos_clean_indices, class_idx] = 1
            labels[pos_noisy_indices, class_idx] = 0  # Hard correction for noisy samples

            # Soft correction for uncertain samples
            labels[pos_uncertain_indices, class_idx] = (
                soft_correction_weight * pred_labels[pos_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - soft_correction_weight) * labels[pos_uncertain_indices, class_idx]
            ).to(labels.dtype)

            print("Positive clean indices:", pos_clean_indices.sum().item())
            print("Positive uncertain indices:", pos_uncertain_indices.sum().item())

        if len(neg_class_selection_metric) > 1:
            # Fit GMM for negative samples in this class
            neg_class_sm_np = neg_class_selection_metric.detach().cpu().numpy()
            neg_gmm = fit_gaussian_model(neg_class_sm_np)
            neg_clean_mean, neg_noisy_mean = np.sort(neg_gmm.means_.flatten())
            print("Negative GMM Means:", neg_gmm.means_.flatten())

            ############################## NEGATIVE CORRECTION #############################
            neg_clean_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] <= neg_clean_mean) 
            neg_uncertain_indices = class_neg_pair_indices & class_neg_indices & (neg_clean_mean < neg_selection_metric[:, class_idx]) & (neg_selection_metric[:, class_idx] <= neg_noisy_mean) 
            neg_noisy_indices = class_neg_pair_indices & class_neg_indices & (neg_selection_metric[:, class_idx] > neg_noisy_mean) 

            clean_mask[neg_clean_indices, class_idx] = 1  # Mark negatives as clean

            labels[neg_clean_indices, class_idx] = 0
            labels[neg_noisy_indices, class_idx] = 1  # Hard correction for noisy samples

            # Soft correction for uncertain samples
            labels[neg_uncertain_indices, class_idx] = (
                soft_correction_weight * pred_labels[neg_uncertain_indices, class_idx].to(labels.dtype) +
                (1 - soft_correction_weight) * labels[neg_uncertain_indices, class_idx]
            ).to(labels.dtype)

            print("Negative clean indices:", neg_clean_indices.sum().item())
            print("Negative uncertain indices:", neg_uncertain_indices.sum().item())
            
        
    labels = torch.tensor(labels, device = pred_labels.device)
        
    # Compute correction loss
    corr_loss = criterion_now_each(pred_labels, labels)
    corr_loss = corr_loss.mean()
    
    
    
    
    
    
    return corr_loss







import torch
import torch.nn as nn

def compute_correlation_matrix(labels):
    """
    Compute the correlation matrix for the given labels.

    Args:
        labels (torch.Tensor): Tensor of shape (batch_size, num_labels) with binary or continuous values.

    Returns:
        torch.Tensor: Correlation matrix of shape (num_labels, num_labels).
    """
    # Compute mean for each label
    mean_labels = labels.mean(dim=0, keepdim=True)
    
    # Compute centered labels
    centered_labels = labels - mean_labels
    
    # Compute covariance matrix
    covariance_matrix = centered_labels.T @ centered_labels / (labels.shape[0] - 1)
    
    # Compute standard deviations
    stddev = torch.sqrt(torch.diag(covariance_matrix)).unsqueeze(0)
    
    # Avoid division by zero
    stddev = torch.where(stddev == 0, torch.ones_like(stddev), stddev)
    
    # Compute correlation matrix
    correlation_matrix = covariance_matrix / (stddev.T @ stddev)
    return correlation_matrix

class MultiLabelCorrelationLoss(nn.Module):
    """
    Multi-label correlation loss that aligns predicted and true correlation matrices.

    Args:
        reduction (str): Reduction method ('mean' or 'sum').
    """
    def __init__(self, reduction='mean'):
        super(MultiLabelCorrelationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Compute the correlation loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities of shape (batch_size, num_labels).
            y_true (torch.Tensor): Ground truth labels of shape (batch_size, num_labels).

        Returns:
            torch.Tensor: Correlation loss value.
        """
        # Compute ground truth and predicted correlation matrices
        corr_true = compute_correlation_matrix(y_true)
        corr_pred = compute_correlation_matrix(y_pred)
        
        # Compute the Frobenius norm of the difference
        loss = torch.norm(corr_true - corr_pred, p='fro')
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss / y_true.size(1)  # Normalize by number of labels
        elif self.reduction == 'sum':
            pass  # Keep the sum as is

        return loss




def compute_class_weights(noisy_target):
    """
    Compute class weights based on binary noisy target data for multi-label classification.
    
    Args:
        noisy_target (torch.Tensor): Binary matrix of noisy targets (128 x 25).
    
    Returns:
        torch.Tensor: Class weights for each label (size: 25).
    """
    # Ensure the input is a tensor
    if not isinstance(noisy_target, torch.Tensor):
        noisy_target = torch.tensor(noisy_target, dtype=torch.float32)

    # Move to CPU if needed
    noisy_target = noisy_target.detach().cpu()

    # Sum the positive occurrences for each class (column-wise)
    class_counts = noisy_target.sum(dim=0)  # Shape: (25,)

    # Total number of samples
    total_samples = noisy_target.size(0)

    # Compute class weights inversely proportional to positive counts
    # Avoid division by zero with a small epsilon
    epsilon = 1e-6
    class_weights = total_samples / (class_counts + epsilon)

    # Normalize weights to ensure they sum to the number of classes
    class_weights = class_weights / class_weights.sum() * noisy_target.size(1)

    return torch.tensor(class_weights, dtype=torch.float32).to(device)





def compare_rank(rank_all, clean_labels, labels, x_ids, epoch):
    # Stage 1:  Select top k indices for each label (1-25) for both 1 and 0 class    
    labels_numpy = labels.detach().cpu().numpy().squeeze() 
    clean_labels_numpy = clean_labels.detach().cpu().numpy().squeeze() 
    
    curr_rank_all = rank_all[x_ids,:,epoch]
    rank_all_flat = curr_rank_all.reshape(-1)
    
    # Positive Pair
    clean_metrics = curr_rank_all[clean_labels_numpy == labels_numpy]
    noisy_metrics = curr_rank_all[clean_labels_numpy != labels_numpy]
    
    
    print("RANK METRIC FOR CLEAN AND NOISY SET")
    stats = compare_selection_metric(rank_all_flat, clean_metrics, noisy_metrics)
    
    print(stats)
    

def compare_loss(loss_all, clean_labels, labels, x_ids, epoch):
    # Stage 1:  Select top k indices for each label (1-25) for both 1 and 0 class    
    labels_numpy = labels.detach().cpu().numpy().squeeze() 
    clean_labels_numpy = clean_labels.detach().cpu().numpy().squeeze() 
    
    curr_loss_all = loss_all[x_ids,:,epoch]
    loss_all_flat = curr_loss_all.reshape(-1)
    
    # Positive Pair
    clean_metrics = curr_loss_all[clean_labels_numpy == labels_numpy]
    noisy_metrics = curr_loss_all[clean_labels_numpy != labels_numpy]
    
    print("LOSS METRIC FOR CLEAN AND NOISY SET")
    stats = compare_selection_metric(loss_all_flat, clean_metrics, noisy_metrics)
    print(stats)
    

def compare_selection_metric(selection_metric, clean_metrics, noisy_metrics):
    """
    Compare selection metrics for clean and noisy labels.
    
    Args:
        selection_metric (torch.Tensor): Selection metric, shape (num_samples, num_labels).
        clean_label (torch.Tensor): Binary clean labels, shape (num_samples, num_labels).
        noisy_label (torch.Tensor): Binary noisy labels, shape (num_samples, num_labels).
    
    Returns:
        dict: Summary statistics for clean and noisy labels.
    """
    # Summary statistics for PyTorch tensors
    if isinstance(clean_metrics, torch.Tensor) and isinstance(noisy_metrics, torch.Tensor):
        # Summary statistics for PyTorch tensors
        stats = {
            "clean": {
                "mean": torch.mean(clean_metrics).item(),
                "std": torch.std(clean_metrics).item(),
                "min": torch.min(clean_metrics).item(),
                "max": torch.max(clean_metrics).item(),
                "count": clean_metrics.numel(),
            },
            "noisy": {
                "mean": torch.mean(noisy_metrics).item(),
                "std": torch.std(noisy_metrics).item(),
                "min": torch.min(noisy_metrics).item(),
                "max": torch.max(noisy_metrics).item(),
                "count": noisy_metrics.numel(),
            },
        }
    elif isinstance(clean_metrics, np.ndarray) and isinstance(noisy_metrics, np.ndarray):
        # Summary statistics for NumPy arrays
        stats = {
            "clean": {
                "mean": np.mean(clean_metrics),
                "std": np.std(clean_metrics),
                "min": np.min(clean_metrics),
                "max": np.max(clean_metrics),
                "count": len(clean_metrics),
            },
            "noisy": {
                "mean": np.mean(noisy_metrics),
                "std": np.std(noisy_metrics),
                "min": np.min(noisy_metrics),
                "max": np.max(noisy_metrics),
                "count": len(noisy_metrics),
            },
        }
    else:
        raise TypeError("Metrics must be either PyTorch tensors or NumPy arrays.")
    return stats




def calculate_confidence_simplified(y_pred, y_noisy):
    """
    Calculate confidence scores for each sample based on predictions and noisy labels (simplified).
    
    Args:
        y_pred (torch.Tensor): Predicted probabilities for positive class (N x K).
        y_noisy (torch.Tensor): Noisy labels (N x K).
        
    Returns:
        torch.Tensor: Confidence scores for each instance (N,).
    """
    # Separate positive and negative masks
    pos_mask = (y_noisy == 1)
    neg_mask = (y_noisy == 0)
    
    # Compute P(k) and A(k) using masked mean
    P_k = (y_pred * pos_mask).sum(dim=0) / pos_mask.sum(dim=0).clamp(min=1)
    A_k = ((1 - y_pred) * neg_mask).sum(dim=0) / neg_mask.sum(dim=0).clamp(min=1)
    
    # Compute confidence scores
    confidence_scores = (y_noisy * P_k + (1 - y_noisy) * A_k).sum(dim=1)
    
    
    # Ensure no division by zero
    epsilon = 1e-8
    inv_scores = 1 / (confidence_scores + epsilon)
    
    # Normalize to get valid probabilities
    sampling_probs = inv_scores / inv_scores.sum()

    return confidence_scores, sampling_probs






def augment_top_clean_positive_instances(
    ehr, ehr_length, mask_ehr, note, mask_note, y_noisy, y_pred, clean_mask, num_instances=5):
    """
    Select top instances with the highest clean positive entries and perform augmentation.

    Args:
        ehr (torch.Tensor): EHR data (batch_size, ...).
        ehr_length (torch.Tensor): Length of each EHR sequence (batch_size,).
        mask_ehr (torch.Tensor): EHR masks (batch_size, ...).
        note (torch.Tensor): Note data (batch_size, ...).
        mask_note (torch.Tensor): Note masks (batch_size, ...).
        y_noisy (torch.Tensor): Labels (batch_size, num_classes).
        clean_mask (torch.Tensor): Clean mask indicating clean entries (batch_size, num_classes).
        num_instances (int): Number of top instances to augment. Default: 5.

    Returns:
        Augmented tensors (ehr_aug, ehr_length_aug, mask_ehr_aug, note_aug, mask_note_aug, y_aug).
    """
    # Find clean positive entries (clean_mask == 1 and y_noisy == 1)
    # Ensure clean_mask and y_noisy are boolean tensors
    clean_mask_bool = clean_mask.bool()
    y_noisy_bool = y_noisy.bool()

    # Perform the bitwise AND operation
    clean_positive_entries = (clean_mask_bool & (y_noisy_bool == True)).sum(dim=1)
    
    # Get indices of the top instances with the highest clean positive entries
    # Number of clean positive sample number as first selection criteria
    score, prob = calculate_confidence_simplified(y_pred, y_noisy)
    top_indices = torch.topk(clean_positive_entries, k=num_instances//2, largest=True).indices
    top_indices = torch.tensor(top_indices, dtype=torch.long, device=y_noisy.device)
    
    
    # Score perspective
    score = score.squeeze()  # Shape becomes (128,)
    top_score_indices = torch.topk(score, k=num_instances, largest=True).indices
    top_score_indices = torch.tensor(top_indices, dtype=torch.long, device=y_noisy.device)
    
    
    # Combine both tensors and find the unique values for the union
    union_indices = torch.cat((top_indices, top_score_indices)).unique()

    # Select the top instances
    # print("EHR Top Shape:", ehr_top.shape)

    ehr_len_list = [ehr_length[i] for i in union_indices.tolist()]
    ehr_aug = ehr[union_indices,:max(ehr_len_list),:]
    # print("EHR Length List:", ehr_len_list)

    mask_ehr_aug = mask_ehr[union_indices]
    # print("Mask EHR Aug Shape:", mask_ehr_aug.shape)

    note_aug = [note[i] for i in union_indices.tolist()]
    # print("Note Aug:", note_aug)

    mask_note_aug = mask_note[union_indices]
    # print("Mask Note Aug Shape:", mask_note_aug.shape)
    


    # ehr_aug, note_aug = augment_inst(ehr_aug, note_aug)
    
    
    return union_indices, ehr_aug, ehr_len_list, mask_ehr_aug, note_aug, mask_note_aug





import numpy as np
from tsaug import TimeWarp

import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random



import numpy as np
from tsaug import TimeWarp
from textattack.augmentation import WordNetAugmenter
import torch

# Create augmenters
ehr_augmenter = TimeWarp(n_speed_change=3, max_speed_ratio=3)  # TimeWarp for EHR
text_augmenter = WordNetAugmenter()  # TextAttack's WordNet Augmenter for text



def pad_to_length(tensor, length, pad_value=0.0):
    """
    Pad or truncate a tensor to a fixed length along the time dimension.

    :param tensor: Tensor of shape (batch_size, time_steps, features).
    :param length: Desired fixed length for the time dimension.
    :param pad_value: Value to use for padding. Default is 0.0.
    :return: Padded or truncated tensor of shape (batch_size, length, features).
    """
    batch_size, time_steps, features = tensor.shape
    if time_steps < length:
        pad_size = length - time_steps
        tensor = F.pad(tensor, (0, 0, 0, pad_size), value=pad_value)
    else:
        tensor = tensor[:, :length, :]
    return tensor

def augment_inst(ehr, note, max_length=513):
    """
    Augment EHR and note data, and pad or truncate to fixed length.
    
    :param ehr: A torch tensor of shape (batch_size, time_steps, features).
    :param note: A batch of note data (list of strings).
    :param max_length: Fixed length for time dimension in EHR and notes.
    :return: Augmented and padded EHR and note data.
    """
    # Convert EHR tensor to numpy for augmentation
    ehr_np = ehr.cpu().numpy()

    # Augment EHR data with TimeWarping
    augmented_ehr = []
    for idx, series in enumerate(ehr_np):
        augmented_series = ehr_augmenter.augment(series)
        augmented_ehr.append(augmented_series)

    # Convert back to torch tensor
    augmented_ehr = torch.tensor(augmented_ehr, dtype=torch.float32).to(ehr.device)


    # Augment Note data with TextAttack's WordNet Augmenter
    augmented_note = [text_augmenter.augment(sentence)[0] if sentence else "" for sentence in note]
    
    
    
    # print("Original Notes:")
    # for i, sentence in enumerate(note):
    #     length = len(sentence.split())  # Split sentence into words and count them
    #     print(f"Sentence {i + 1}: Length = {length}, Content = {sentence}")

    # print("\nAugmented Notes:")
    # for i, sentence in enumerate(augmented_note):
    #     length = len(sentence.split())  # Split sentence into words and count them
    #     print(f"Sentence {i + 1}: Length = {length}, Content = {sentence}")
        
        
    
    return augmented_ehr, augmented_note





def apply_label_smoothing(labels, alpha=0.1):
    """
    Apply label smoothing for binary labels.

    Args:
        labels (torch.Tensor): Original labels, shape (n_samples, n_classes).
        alpha (float): Smoothing factor.

    Returns:
        torch.Tensor: Smoothed labels.
    """
    smoothed_labels = (1 - alpha) * labels + alpha / 2  # Binary smoothing
    return smoothed_labels