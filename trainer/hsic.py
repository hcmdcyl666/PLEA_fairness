"""
Original code:
    https://github.com/clovaai/rebias
"""
import torch
import torch.nn as nn

def to_numpy(x):
    """convert Pytorch tensor to numpy array
    """
    return x.clone().detach().cpu().numpy()

class HSIC(nn.Module):
    """Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.
    Empirically, we use the finite sample estimator of HSIC (with m observations) by,
    (1) biased estimator (HSIC_0)
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        :math: (m - 1)^2 tr KHLH.
        where K_{ij} = kernel_x (x_i, x_j), L_{ij} = kernel_y (y_i, y_j), H = 1 - m^{-1} 1 1 (Hence, K, L, H are m by m matrices).
    (2) unbiased estimator (HSIC_1)
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        :math: \frac{1}{m (m - 3)} \bigg[ tr (\tilde K \tilde L) + \frac{1^\top \tilde K 1 1^\top \tilde L 1}{(m-1)(m-2)} - \frac{2}{m-2} 1^\top \tilde K \tilde L 1 \bigg].
        where \tilde K and \tilde L are related to K and L by the diagonal entries of \tilde K_{ij} and \tilde L_{ij} are set to zero.
    Parameters
    ----------
    sigma_x : float
        the kernel size of the kernel function for X.
    sigma_y : float
        the kernel size of the kernel function for Y.
    algorithm: str ('unbiased' / 'biased')
        the algorithm for the finite sample estimator. 'unbiased' is used for our paper.
    reduction: not used (for compatibility with other losses).
    """
    def __init__(self, sigma_x, sigma_y=None, algorithm='unbiased',
                 reduction=None):
        super(HSIC, self).__init__()

        if sigma_y is None:
            sigma_y = sigma_x

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        if algorithm == 'biased':
            self.estimator = self.biased_estimator
        elif algorithm == 'unbiased':
            self.estimator = self.unbiased_estimator
        else:
            raise ValueError('invalid estimator: {}'.format(algorithm))

    def _kernel_x(self, X):
        raise NotImplementedError

    def _kernel_y(self, Y):
        raise NotImplementedError

    def biased_estimator(self, input1, input2):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        K = self._kernel_x(input1)
        L = self._kernel_y(input2)

        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)

        N = len(input1)

        return torch.trace(KH @ LH / (N - 1) ** 2)

    def unbiased_estimator(self, input1, input2):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        N = len(input1) # N起码要大于等于4吧
        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        return hsic / (N * (N - 3))
    
    def unbiased_estimator_1(self, input1, input2, w_y):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        kernel_XX = self._kernel_x(input1)
        # kernel_XX/kernel_YY是一个64x64维的矩阵，其中(i, j)处的元素代表了第i个样本和第j个样本映射到高维空间后的内积。
        # w_y = torch.exp(10*w_y)
        # kernel_YY = self._kernel_y(input2, w_y)
        kernel_YY = self._kernel_y(input2)

        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        N = len(input1)
        hsic = (
            torch.trace(tK @ tL)
            # + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            + (torch.sum(tK) * torch.sum(tL) / (N) / (N)) # N有可能为1或者2或者3
            # - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N))
        )
        # return hsic / (N * (N - 3))
        return hsic / (N * (N))

    def unbiased_estimator_with_weights_adjust(self, input1, input2, weights):

        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device='cuda')
        # weight_sum = torch.sum(weights_tensor)
        # N = weights_tensor.size(0)  # 获取样本数量
        # # weight_sum = N # 直接用样本数量
        # sqrt_weights = torch.sqrt(weights_tensor)

        # 对权重进行指数调整(α=0、1、1.5、2、3、4、5)
        # exp_weights_tensor = torch.exp(weights_tensor)
        exp_weights_tensor = weights_tensor
       
        weight_sum = torch.sum(exp_weights_tensor)
        N = exp_weights_tensor.size(0)  # 获取样本数量

        sqrt_weights = torch.sqrt(exp_weights_tensor)  # 使用调整后的权重
        norm_factor = torch.outer(sqrt_weights, sqrt_weights)
        weighted_K = kernel_XX * norm_factor
        weighted_L = kernel_YY * norm_factor
        tK = weighted_K - torch.diag(torch.diag(weighted_K))
        tL = weighted_L - torch.diag(torch.diag(weighted_L))
        # 确保不会出现除以接近0的数
        epsilon = 1e-6
        # weight_sum_adjusted = max(weight_sum - 1, epsilon) * max(weight_sum - 2, epsilon) * max(weight_sum - 3, epsilon)
        weight_sum_adjusted = max(weight_sum  * (weight_sum - 3), epsilon) # 和上面unbias函数最后的除项对齐

        # hsic_numerator = torch.trace(tK @ tL) + (torch.sum(tK) * torch.sum(tL) / (max((weight_sum - 1, epsilon) * max(weight_sum - 2, epsilon)))
        hsic_numerator = torch.trace(tK @ tL) + (torch.sum(tK) * torch.sum(tL) / (max((weight_sum - 1) * (weight_sum - 2), epsilon)))
        hsic_denominator = 2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / max(weight_sum - 2, epsilon)
        hsic = (hsic_numerator - hsic_denominator) / weight_sum_adjusted

        return hsic

    def unbiased_estimator_with_weights(self, input1, input2, weights):
        kernel_XX = self._kernel_x(input1)  # 假设返回的是一个NxN的矩阵
        kernel_YY = self._kernel_y(input2)  # 同上，NxN的矩阵
        weights_tensor = torch.tensor(weights, dtype=torch.float32)  # 转换权重为张量
        N = weights_tensor.size(0)  # 获取样本数量
        weight_sum = torch.sum(weights_tensor)  # 权重总和
        N = weight_sum
        sqrt_weights = torch.sqrt(weights_tensor)
        norm_factor = torch.outer(sqrt_weights, sqrt_weights)
        weighted_K = kernel_XX * norm_factor  # 按元素乘法调整权重
        weighted_L = kernel_YY * norm_factor
        # 计算HSIC值
        tK = weighted_K - torch.diag(torch.diag(weighted_K))
        tL = weighted_L - torch.diag(torch.diag(weighted_L))

        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        ) / (N * (N - 3))

        return hsic

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)

class RbfHSIC(HSIC):
    """Radial Basis Function (RBF) kernel HSIC implementation.
    """
    def _kernel_raw(self, X, sigma):
        X = X.view(len(X), -1)
        Xn = X.norm(2, dim=1, keepdim=True)
        X = X.div(Xn)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        X_L2 = X_L2.clamp(1e-12)
        sigma_avg = X_L2.mean().detach()
        gamma = 1/(2*sigma_avg)
        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def _kernel(self, X, sigma, w=None):
        # 添加一个小的常数防止除以零
        # 因为特征X可能出现nan值
        epsilon = 1e-10
        X = X + epsilon
        X = X.view(len(X), -1) # 加了weight之后 X有可能会含有nan
        if w is not None:
            # w_sqrt = w.sqrt().unsqueeze(1)  # 计算权重的平方根，并给它添加一个维度以匹配X的形状
            X = X * w.unsqueeze(1)  # 应用权重到X的每个样本
        Xn = X.norm(2, dim=1, keepdim=True)
        
        # Xn = Xn + epsilon

        if torch.any(Xn == 0):
            print("Zero detected in Xn after adjustment")

        X = X.div(Xn)

        if torch.any(torch.isnan(X)):
            print("NaN detected in normalized X")

        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)

        if torch.any(torch.isnan(X_L2)):
            print("NaN detected in X_L2 before clamp")

        X_L2 = X_L2.clamp(min=1e-12)

        if torch.any(torch.isnan(X_L2)):
            print("NaN detected in X_L2 after clamp")

        sigma_avg = X_L2.mean().detach()
        gamma = 1 / (2 * sigma_avg)

        kernel_XX = torch.exp(-gamma * X_L2)

        if torch.any(torch.isnan(kernel_XX)):
            print("NaN detected in kernel_XX")

        return kernel_XX

    def _kernel_x(self, X):
        if torch.isnan(X).any():
            print("NaN detected in X")
        return self._kernel(X, self.sigma_x)

    def _kernel_y(self, Y):
        return self._kernel(Y, self.sigma_y)


class MinusRbfHSIC(RbfHSIC):
    """``Minus'' RbfHSIC for the ``max'' optimization.
    """
    def forward(self, input1, input2, **kwargs):
        return -self.estimator(input1, input2)
