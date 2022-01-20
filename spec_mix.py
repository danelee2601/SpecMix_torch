from typing import List, Tuple, Union
import numpy as np
import torch
from torch import Tensor


class SpecMix(object):
    def __init__(self, gamma: float = 0.3, n_mask_bars: int = 3):
        self.gamma = gamma
        self.n_mask_bars = n_mask_bars

    def _select_mask_start_points(self, Sxx: Tensor, f_: int, mask_len: int, kind: str) -> List[int]:
        if kind == 'freq':
            mask_step_ = np.random.randint(0, Sxx.shape[2] + 1 - mask_len, f_)
        elif kind == 'time':
            mask_step_ = np.random.randint(0, Sxx.shape[3] + 1 - mask_len, f_)
        else:
            raise ValueError
        mask_step_ = sorted(mask_step_)
        return mask_step_

    def _get_mask_range(self, mask_step_: List[int], mask_len: int) -> List[Tuple[int, int]]:
        mask_range_ = []
        for m_step in mask_step_:
            mask_range_.append((m_step, m_step + mask_len))
        return mask_range_

    def _clean_mask_range_(self, mask_range_: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        cleaned_mask_range_ = [mask_range_[0]]
        track_range = mask_range_[0]  # init
        prev_added = True
        for i in range(len(mask_range_) - 1):
            if mask_range_[i + 1][0] - track_range[1] > 0:  # if two ranges are not overlapped.
                track_range = mask_range_[i + 1]
                cleaned_mask_range_.append(mask_range_[i + 1])
                prev_added = True
            else:  # if two ranges are overlapped.
                track_range = (track_range[0], mask_range_[i + 1][1])
                if prev_added:
                    cleaned_mask_range_.pop(-1)
                    prev_added = False
                if i == len(mask_range_) - 2:
                    cleaned_mask_range_.append(track_range)
        return cleaned_mask_range_

    def __call__(self, Sxx: Tensor, labels: Tensor):
        """
        :param Sxx: (batch_size, channel_size, height, width)
        :param labels: (batch_size,)
        :return:
        """

        # select `f_freq` and `f_time` for the entire mini-batch, B_1.
        f_freq = np.random.randint(0, self.n_mask_bars + 1)
        f_time = np.random.randint(0, self.n_mask_bars + 1)

        # get a randomly-permuted (rp) mini-batch, `B_2`
        rand_index = torch.randperm(Sxx.size()[0])
        Sxx_rp = Sxx[rand_index, :, :, :]
        labels_rp = labels[rand_index]
        # print('Sxx.shape:', Sxx.shape)

        # compute mask length for each masking
        mask_height = np.floor(Sxx.shape[2] * self.gamma).astype(int)
        mask_width = np.floor(Sxx.shape[3] * self.gamma).astype(int)
        # print('mask_height:', mask_height)
        # print('mask_width:', mask_width)

        # compute masking steps in y-axis (freq-axis) and x-axis (time-axis)
        mask_step_f = self._select_mask_start_points(Sxx, f_freq, mask_height, 'freq')
        mask_step_t = self._select_mask_start_points(Sxx, f_time, mask_width, 'time')
        # print('mask_step_f:', mask_step_f)
        # print('mask_step_t:', mask_step_t)

        # compute masking ranges in y-axis (freq-axis) and x-axis (time-axis)
        mask_range_f = self._get_mask_range(mask_step_f, mask_height)
        mask_range_t = self._get_mask_range(mask_step_t, mask_width)
        # print('mask_range_f:', mask_range_f)
        # print('mask_range_t:', mask_range_t)

        # mask in y-axis (freq-axis)
        Sxx_rp_ = torch.zeros(Sxx_rp.shape).float().to(Sxx.device)
        if mask_range_f:
            cleaned_mask_range_f = self._clean_mask_range_(mask_range_f)
            # print('cleaned_mask_range_f:', cleaned_mask_range_f)
            # mask - freq
            for rng in cleaned_mask_range_f:
                Sxx[:, :, rng[0]:rng[1], :] = 0.
                Sxx_rp_[:, :, rng[0]:rng[1], :] = Sxx_rp[:, :, rng[0]:rng[1], :]

        # mask in x-axis (time-axis)
        if mask_range_t:
            cleaned_mask_range_t = self._clean_mask_range_(mask_range_t)
            # print('cleaned_mask_range_t:', cleaned_mask_range_t)
            # mask - time
            for rng in cleaned_mask_range_t:
                Sxx[:, :, :, rng[0]:rng[1]] = 0.
                Sxx_rp_[:, :, :, rng[0]:rng[1]] = Sxx_rp[:, :, :, rng[0]:rng[1]]

        # mix as Eq. (1) in the original paper
        mixed_Sxx = (Sxx + Sxx_rp_).float()

        # compute `lambda` as in Eq. (2)
        masked_size = (Sxx[0][0] != 0).int().sum()
        img_size = Sxx[0][0].shape[0] * Sxx[0][0].shape[1]
        lambda_ = masked_size / img_size
        # print('masked_size:', masked_size)
        # print('img_size:', img_size)
        # print('lambda_:', lambda_)
        # print('\n\n')
        return mixed_Sxx, lambda_, labels, labels_rp


if __name__ == '__main__':
    import random
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    random.seed(2)
    np.random.seed(1)
    torch.manual_seed(0)

    # toy dataset
    batch_size = 8
    channel_size = 1
    height = 224
    width = 224
    Sxx = torch.rand((batch_size, channel_size, height, width)).abs()  # spectrogram (Sxx)
    scale = torch.rand((batch_size, 1, 1, 1))
    Sxx = Sxx * scale
    labels = torch.randint(0, 1, (batch_size, ))

    # apply SpecMix
    spec_mix = SpecMix()
    mixed_Sxx, lambda_, labels, labels_rp = spec_mix(Sxx, labels)

    # loss would look like:
    # loss = lambda_ * criterion(pred_labels, labels) + (1 - lambda_) * criterion(pred_labels, labels)

    # take sample
    mixed_sxx = mixed_Sxx[0][0]

    # plot
    fig, ax1 = plt.subplots(figsize=(4, 3))
    im1 = ax1.imshow(mixed_sxx, aspect='auto')
    ax1.invert_yaxis()
    fig.colorbar(im1)
    plt.show()
