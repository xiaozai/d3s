import torch
from pytracking import optimization, TensorList, operation
import math


class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation, response_activation, training_samples_d=None, compression=True):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

        # Song ,
        self.training_samples_d = training_samples_d # Song
        self.compression = compression

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        filter = x[:len(x)//2]  # w2 in paper, x[0]
        P = x[len(x)//2:]       # w1 in paper, x[1]

        # Do first convolution
        if self.compression:
            compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)
        else:
            compressed_samples = self.training_samples

        print('Song : in FactorizedConvProblem: compressed_samples:', compressed_samples[0].shape, self.training_samples_d.shape)
        # Do second convolution,
        residuals = operation.conv2d(compressed_samples, filter, mode='same', depth=self.training_samples_d).apply(self.response_activation)

        # Compute data residuals
        residuals = residuals - self.y

        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)

        # Add regularization for projection matrix
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals


    def ip_input(self, a: TensorList, b: TensorList):
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        return x / self.diag_M


class ConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, sample_weights: TensorList, response_activation,
                training_samples_d=None):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

        self.training_samples_d = training_samples_d

    def __call__(self, x: TensorList, depth=None):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]

        Song, add depthconv, does not work, since it requires pytorch 0.4.0
        """
        # Do convolution and compute residuals
        print('self.training_samples in ConvProblem: ', self.training_samples.shape, self.training_samples_d.shape)
        residuals = operation.conv2d(self.training_samples, x, mode='same', depth=self.training_samples_d).apply(self.response_activation)
        residuals = residuals - self.y

        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)
