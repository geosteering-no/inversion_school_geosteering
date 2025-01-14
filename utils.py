import random
import torch

random.seed(0)
torch.manual_seed(0)

def convert_to_image(reference_log, my_log):
    # ref, data = x
    # ref_mat = ref.repeat(1, self.input_data_size,1)
    ref2 = reference_log.unsqueeze(2)
    input_data_size = my_log.size()[1]
    ref_mat = ref2.repeat((1, 1, input_data_size))
    data1 = my_log.unsqueeze(1)
    ref_data_size = reference_log.size()[1]
    data_mat = data1.repeat(1, ref_data_size, 1)
    image = 0.5 + (ref_mat - data_mat) * 0.5
    image = image.unsqueeze(1)
    return image

def default_traj_to_index(self, traj, device):
    half = self.input_ref_size // 2
    index = traj * half + half
    index_trunc = torch.maximum(index, torch.ones(index.size(), device=device))
    index_trunc = torch.minimum(index_trunc,
                                torch.ones(index.size(), device=device) * (self.input_ref_size-2))
    return index_trunc

def evaluate_log(self, ref_data, output, my_arange, device):
    # todo refactor to remove self
    my_arange_uns = my_arange.unsqueeze(1)
    my_arange_uns = my_arange_uns.unsqueeze(2)
    #TODO fixme should be equal to the inversion size not prediciton size
    first_index = my_arange_uns.repeat(1, self.modes, self.input_data_size)
    expanded = self._expand(output)[:, :, 0:self.input_data_size]
    expanded_trunc = self.default_traj_to_index(expanded, device)
    indexes = (expanded_trunc + 0.5).long()
    i0s = (expanded_trunc).floor().detach()
    i1s = i0s + 1
    # indexes = torch.max(indexes, 0)
    # indexes = torch.min(indexes, self.input_ref_size-1)
    curves_values0 = ref_data[first_index, i0s.long()]
    curves_values1 = ref_data[first_index, i1s.long()]
    dists0 = expanded_trunc - i0s
    dists1 = i1s - expanded_trunc
    curves_values = dists1 * curves_values0 + dists0 * curves_values1
    return curves_values


