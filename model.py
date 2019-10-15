import torch.nn as nn
from modules import glimpse_network, decision_network
from modules import action_network, illumination_network
from torch.nn import LSTMCell


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """
    def __init__(self,
                 channels,
                 h_g,
                 h_l,
                 std,
                 hidden_size,
                 num_classes,
                 learned_start):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - c: number of num_channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of num_classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        self.std = std

        self.sensor = glimpse_network(h_g, h_l, learned_start, channels)
        self.rnn = LSTMCell(256, hidden_size)
        self.decision = decision_network(hidden_size, 2)
        self.illuminator = illumination_network(hidden_size, channels, std)
        self.classifier = action_network(hidden_size, num_classes)

    def forward(self, x, k_t_prev, h_t_prev, last=False, valid=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the num_classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and them
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the num_classes.
        - log_pi: a vector of length (B,).
        """
        # sample k-space
        g_t = self.sensor(x, k_t_prev)
        h_t = self.rnn(g_t, h_t_prev)
        # h_t = self.rnn(g_t, h_t_prev)
        # if h_t_prev is None:
        #     self.h_t2_prev = None
        # h_t = self.rnn_2(h_t[0], self.h_t2_prev)
        # self.h_t2_prev = h_t

        mu = self.illuminator(h_t[0], valid)
        d, log_d = self.decision(h_t[0])

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        # log_pi = Normal(mu, self.std).log_prob(k_t)
        # log_pi = torch.sum(log_pi, dim=1)
        log_probas = self.classifier(h_t[0])
        return h_t, mu, log_probas, d, log_d

