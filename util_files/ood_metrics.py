import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def compute_energy(logits, T=1.0):
    """
    Compute the energy score for given logits and a temperature scaling factor T.
    """
    # 计算log-sum-exp来得到能量分数
    # 能量值越小，样本越可能属于训练数据的分布
    max_logits = torch.max(logits, dim=1, keepdim=True).values  # 稳定数值的技巧，防止exp溢出
    energy = -T * torch.log(torch.sum(torch.exp((logits - max_logits) / T), dim=1)) - max_logits.squeeze(1)
    return energy
def energy_distance_with_indices(Y_logits, Y_indices, G_pred, G_prob, G_Y):
    # Y_logits中每个元素是一个样本的logits（两类就是两个值）
    # 初始化返回的字典
    distances_dict = {}
    energys = compute_energy(Y_logits)
    # 计算Y中每个样本到它对应组的X的马氏距离，并检查预测正确性
    for i, e in enumerate(energys):
        group = G_pred[i]
        # prob = G_prob[i].max()
        real_group = G_Y[i]
        # energy = compute_energy(logits)  # Compute the energy score
        # 检查组别预测是否正确
        prediction_correct = 1 if group == real_group else 0
        # 使用Y_indices[i]作为键，[马氏距离, 预测正确性]作为值更新字典
        # 使用Y_indices[i]作为键，[马氏距离, 预测概率最大值, 预测正确性]作为值更新字典
        # distances_dict[Y_indices[i]] = [round(e.item(), 2), round(prob.item(),2), prediction_correct]
        distances_dict[Y_indices[i]] = round(e.item(), 2)
    return distances_dict

def gram_matrix(features):
    # features: Tensor of shape (N, D) where N is the number of samples and D is the number of features
    return torch.mm(features, features.t()) / features.size(1)
def gram_distance_with_indices(X, Y, Y_indices, G_X, G_pred, G_Y):
    # 初始化返回的字典
    distances_dict = {}
    
    # 为X计算Gram矩阵
    X_gram = gram_matrix(X)
    
    # 计算Y中每个样本与X的Gram矩阵的差异
    for i, y in enumerate(Y):
        # 计算包含当前Y样本的新矩阵的Gram矩阵
        Y_extended = torch.cat([X, y.unsqueeze(0)], dim=0)
        Y_gram = gram_matrix(Y_extended)
        
        # 计算最后一行（与Y样本相关）与X的Gram矩阵的行的差异
        gram_differences = Y_gram[-1, :-1] - X_gram.mean(dim=0)
        
        # 使用L2范数作为差异度量
        distance = torch.norm(gram_differences, p=2)
        
        # 检查组别预测是否正确
        prediction_correct = 1 if G_pred[i] == G_Y[i] else 0
        
        # 使用Y_indices[i]作为键，[Gram距离, 预测正确性]作为值更新字典
        distances_dict[Y_indices[i]] = [round(distance.item(), 2), prediction_correct]
    
    return distances_dict
def calculate_similarity_means(X, Y, Y_indices, G_pred, G_Y, G_prob):
    # 计算余弦相似度和欧氏距离
    cosine_sim = cosine_similarity(Y, X)  # 形状为[len(Y), len(X)]
    euclidean_dist = euclidean_distances(Y, X)  # 形状为[len(Y), len(X)]
    # 计算均值
    cosine_means = cosine_sim.mean(axis=1)  # 对每个Y样本，计算与所有X样本的余弦相似度均值
    euclidean_means = euclidean_dist.mean(axis=1)  # 对每个Y样本，计算与所有X样本的欧氏距离均值
    # 创建字典，包含每个样本的均值
    results = {Y_indices[i]: [round(cosine_means[i],2), round(euclidean_means[i],2), round(G_prob[i].max().item(),2)] for i in range(len(Y_indices))}
    return results

def mahalanobis_distance_with_indices_unique_groups(X, Y, neighber_num, Y_indices, G_X, G_prob, G_Y, regularization=1e-6):
    # X是一个n*d的Tensor，表示数据集
    # Y是一个m*d的Tensor，表示另一组样本
    # Y_indices是Y中每个样本的索引列表或Tensor
    # G_X是一个长度为n的Tensor，表示X中每个样本的组别（0或1）
    # G_pred是一个长度为m的Tensor，表示对于Y中每个样本预测的组别
    # G_Y是一个长度为m的Tensor，表示Y中每个样本的真实组别
    # regularization是正则化项的系数
    
    # 初始化返回的字典
    distances_dict = {}
    
    # 为两个组别计算均值和协方差矩阵
    unique_groups = G_X.unique()
    group_stats = {}
    for group in unique_groups:
        group_X = X[G_X == group]
        mean_vector = torch.mean(group_X, axis=0)
        X_np = group_X.detach().cpu().numpy()
        cov_matrix = np.cov(X_np, rowvar=False)
        cov_matrix = torch.from_numpy(cov_matrix).to(X.device)
        cov_matrix += torch.eye(cov_matrix.shape[0], device=cov_matrix.device) * regularization
        cov_matrix_inv = torch.inverse(cov_matrix)
        group_stats[group.item()] = (mean_vector, cov_matrix_inv)
    
    # 计算Y中每个样本到它对应组的X的马氏距离，并检查预测正确性
    for i, y in enumerate(Y):
        # group = G_pred[i]
        prob = G_prob[i].max()
        real_group = G_Y[i]
        mean_vector, cov_matrix_inv = group_stats[group.item()]
        diff_vector = y - mean_vector
        dist_squared = torch.matmul(torch.matmul(diff_vector.unsqueeze(0).float(), cov_matrix_inv.float()), diff_vector.unsqueeze(1).float())
        distance = torch.sqrt(dist_squared)
        # 检查组别预测是否正确
        # prediction_correct = 1 if group == real_group else 0
        # 使用Y_indices[i]作为键，[马氏距离, 预测正确性]作为值更新字典
        distances_dict[Y_indices[i]] = [round(distance.item(), 2), round(prob.item(),2)]
    return distances_dict

def mahalanobis_distance_matrix_with_indices_mm(X, Y, X_indices, regularization=1e-5):
    combined = torch.cat((X, Y), dim=0)
    combined_cpu = combined.cpu().detach()
    cov = torch.Tensor(np.cov(combined_cpu.numpy(), rowvar=False))
    cov_regularized = cov + torch.eye(cov.size(0)) * regularization
    cov_inv = torch.pinverse(cov_regularized).to(X.device)

    distances_dict = {}
    for i, idx in enumerate(X_indices):
        diff = X[i] - Y  # Broadcasting difference
        dist_squared = torch.diag(torch.mm(torch.mm(diff, cov_inv), diff.t()))  # Batch matrix multiplication
        distances = torch.sqrt(dist_squared)
        avg_distance = round(torch.mean(distances).item(), 2)
        distances_dict[int(idx)] = avg_distance

    return distances_dict

def mahalanobis_distance_with_indices(X, Y, Y_indices, regularization=1e-6):
    # X是一个n*d的Tensor，表示数据集
    # Y是一个m*d的Tensor，表示另一组样本
    # Y_indices是Y中每个样本的索引列表或Tensor
    # regularization是正则化项的系数
    
    # 计算X的均值向量和协方差矩阵
    mean_vector = torch.mean(X, axis=0)
    X_np = X.detach().cpu().numpy()  # 转换为NumPy数组以计算协方差矩阵
    cov_matrix = np.cov(X_np, rowvar=False)
    cov_matrix = torch.from_numpy(cov_matrix).to(X.device)
    
    # 添加正则化项以提高数值稳定性
    cov_matrix += torch.eye(cov_matrix.size(0), device=cov_matrix.device) * regularization
    
    cov_matrix_inv = torch.inverse(cov_matrix)
    
    # 初始化返回的字典
    distances_dict = {}
    
    # 计算Y中每个样本到X的马氏距离
    for i, y in enumerate(Y):
        diff_vector = y - mean_vector
        dist_squared = torch.matmul(torch.matmul(diff_vector.unsqueeze(0).float(), cov_matrix_inv.float()), diff_vector.unsqueeze(1).float())
        distance = torch.sqrt(dist_squared)
        # 使用Y_indices[i]作为键，马氏距离作为值更新字典
        distances_dict[Y_indices[i]] = round(distance.item(),2)
    
    return distances_dict

def mahalanobis_distance_matrix_with_indices_old(X, Y, X_indices, regularization=1e-5):
    # print(len(X))
    # print(len(Y))
    # print(len(X_indices))
    # Assuming X_embedding, Y_embedding are torch tensors
    combined = torch.cat((X, Y), dim=0)
    # Move combined tensor to CPU
    combined_cpu = combined.cpu().detach()
    # Calculate covariance matrix
    cov = torch.Tensor(np.cov(combined_cpu.numpy(), rowvar=False))
    # combined = torch.cat((X, Y), dim=0)
    # cov = torch.Tensor(np.cov(combined.numpy(), rowvar=False))
    cov_regularized = cov + torch.eye(cov.size(0)) * regularization
    cov_inv = torch.pinverse(cov_regularized)
    
    # distances_list = []
    distances_list = {}
    for i, idx in enumerate(X_indices):
        distances = torch.zeros(Y.size(0))
        for j in range(Y.size(0)):
            diff = X[i] - Y[j]
            cov_inv = cov_inv.to(diff.device)
            dist_squared = torch.matmul(torch.matmul(diff.unsqueeze(0), cov_inv), diff.unsqueeze(1))
            distances[j] = torch.sqrt(dist_squared)
        avg_distance = round(torch.mean(distances).item(),2)
        distances_list[int(idx)] = avg_distance
    return distances_list

def mahalanobis_distance_matrix_regularized(X, Y, regularization=1e-5):
    """
    Calculate the Mahalanobis distance between each vector in X and each vector in Y
    based on the regularized covariance matrix of the combined data set (X and Y).
    Regularization improves the numerical stability by ensuring the covariance matrix is invertible.
    """
    # Combine X and Y to compute the covariance matrix
    combined = torch.cat((X, Y), dim=0)
    
    # Compute covariance matrix
    cov = torch.Tensor(np.cov(combined.numpy(), rowvar=False))
    
    # Add regularization term to the diagonal of the covariance matrix to ensure it's invertible
    cov_regularized = cov + torch.eye(cov.size(0)) * regularization
    
    # Compute the pseudo-inverse of the regularized covariance matrix
    cov_inv = torch.pinverse(cov_regularized)
    
    # Initialize the distance matrix
    distances = torch.zeros(X.size(0), Y.size(0))
    
    for i in range(X.size(0)):
        for j in range(Y.size(0)):
            diff = X[i] - Y[j]
            dist_squared = torch.matmul(torch.matmul(diff.unsqueeze(0), cov_inv), diff.unsqueeze(1))
            distances[i, j] = torch.sqrt(dist_squared)
    
    # Calculate the average distance for each vector in X
    avg_distances = torch.mean(distances, dim=1)
    return avg_distances

def mahalanobis_distance_matrix_simplified(X, Y):
    """
    Calculate the Mahalanobis distance between each vector in X and each vector in Y.
    Assuming the covariance matrix is the identity matrix for simplification.
    """
    # Number of vectors in X and Y
    num_X = X.size(0)
    num_Y = Y.size(0)

    # Expand X and Y to form the difference matrix
    # X_expanded: [num_X, 1, n] and Y_expanded: [1, num_Y, n]
    X_expanded = X.unsqueeze(1).expand(num_X, num_Y, -1)
    Y_expanded = Y.unsqueeze(0).expand(num_X, num_Y, -1)

    # Difference matrix
    diff_matrix = X_expanded - Y_expanded

    # Calculate the squared Mahalanobis distance
    # Assuming covariance_inverse is the identity matrix
    dist_squared = torch.sum(diff_matrix ** 2, dim=2)

    # Taking the square root gives the Mahalanobis distance
    distances = torch.sqrt(dist_squared)

    # Calculate the average distance for each vector in X
    avg_distances = torch.mean(distances, dim=1)
    return avg_distances

def mahalanobis_distance(x, y):
    """
    Calculate the Mahalanobis distance between two tensors, assuming the covariance matrix is the identity matrix.
    """
    # Assuming the covariance matrix is the identity matrix, so its inverse is also the identity matrix
    covariance_inverse = (torch.eye(x.size(0))).to(dtype=torch.float32).cuda()

    # Calculate the difference vector
    d = (x - y).to(dtype=torch.float32).cuda()
    # d = (x - y).cuda()
    d = d.to(covariance_inverse.device)

    # Calculate the Mahalanobis distance
    distance = torch.sqrt(torch.dot(d, torch.mv(covariance_inverse, d)))
    return distance

###########余弦相似度（Cosine Similarity）/余弦距离（Cosine Distance）
def cosine_distance_mean_tensor(X1, X2):
    """
    Calculate the mean cosine distance of each row in X2 from all rows in X1 for tensors.

    Parameters:
    X1 (torch.Tensor): A tensor (m x k) where each row is a feature vector.
    X2 (torch.Tensor): A tensor (n x k) where each row is a feature vector.

    Returns:
    torch.Tensor: A column vector (n x 1) where each element is the mean cosine distance 
                  of the corresponding row in X2 from all rows in X1.
    """
    # Normalize X1 and X2 rows to unit vectors
    X1_norm = torch.nn.functional.normalize(X1, p=2, dim=1)
    X2_norm = torch.nn.functional.normalize(X2, p=2, dim=1)

    # Calculate cosine similarities
    cosine_similarities = torch.mm(X2_norm, X1_norm.T)

    # Convert similarities to distances
    cosine_distances = 1 - cosine_similarities

    # Calculate the mean cosine distance for each row in X2
    mean_distances = torch.mean(cosine_distances, dim=1, keepdim=True)

    return mean_distances

###########X2中每个样本与X1之间的Wasserstein距离:评估样本与分布之间的差异
import torch
import ot
import numpy as np

def sliced_wasserstein_distance(X1, X2, num_projections=100):
    """
    Calculate the mean sliced Wasserstein distance between samples in X2 and all samples in X1.

    Parameters:
    X1 (torch.Tensor): A tensor (m x k) where each row is a feature vector.
    X2 (torch.Tensor): A tensor (n x k) where each row is a feature vector.
    num_projections (int): The number of random projections for the sliced Wasserstein distance.

    Returns:
    torch.Tensor: A column vector (n x 1) where each element is the mean sliced Wasserstein distance 
                  of the corresponding row in X2 from all rows in X1.
    """
    m, k = X1.shape
    n, _ = X2.shape

    # Convert tensors to numpy arrays for compatibility with the OT library
    X1_np = X1.numpy()
    X2_np = X2.numpy()

    distances = np.zeros(n)

    for _ in range(num_projections):
        # Generate a random projection
        direction = np.random.randn(k)
        direction /= np.linalg.norm(direction)

        # Project the data
        projected_X1 = np.dot(X1_np, direction)
        projected_X2 = np.dot(X2_np, direction)

        # Calculate Wasserstein distance for each projected sample in X2
        for i in range(n):
            distances[i] += ot.emd2([], [], projected_X2[i:i+1], projected_X1)

    # Compute the mean over all projections
    mean_distances = distances / num_projections

    return torch.tensor(mean_distances).unsqueeze(1)

###########X2与X1之间的Wasserstein距离:评估两个分布之间的差异
import numpy as np
import ot
def simplified_sliced_wasserstein_distance(X1, X2, num_projections=100):
    m, k = X1.shape
    n, _ = X2.shape

    X1_np = X1.cpu().numpy()
    X2_np = X2.cpu().numpy()

    distances = np.zeros(n)

    for _ in range(num_projections):
        direction = np.random.randn(k)
        direction /= np.linalg.norm(direction)

        projected_X1 = np.dot(X1_np, direction)
        projected_X2 = np.dot(X2_np, direction)

        for i in range(n):
            # Simplified calculation by computing absolute differences
            distances[i] += np.sum(np.abs(projected_X2[i] - projected_X1))

    mean_distances = distances / num_projections

    return torch.tensor(mean_distances).unsqueeze(1)

def sliced_wasserstein_distance_between_datasets(X1, X2, num_projections=100):
    """
    Calculate the sliced Wasserstein distance between two datasets representing distributions.

    Parameters:
    X1 (numpy.ndarray): A dataset (m x k) where each row is a sample from the distribution.
    X2 (numpy.ndarray): Another dataset (n x k) where each row is a sample from the distribution.
    num_projections (int): Number of random projections for sliced Wasserstein distance.

    Returns:
    float: The sliced Wasserstein distance between the two distributions.
    """
    m, k = X1.shape
    n, _ = X2.shape

    total_distance = 0.0

    for _ in range(num_projections):
        # Generate a random projection
        direction = np.random.randn(k)
        direction /= np.linalg.norm(direction)

        # Project the data
        projected_X1 = np.dot(X1, direction)
        projected_X2 = np.dot(X2, direction)

        # Calculate Wasserstein distance for the projected data
        wasserstein_distance = ot.emd2([], [], projected_X1, projected_X2)
        total_distance += wasserstein_distance

    # Average over all projections
    average_distance = total_distance / num_projections

    return average_distance

# MSP距离


