import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
from .homography import generate_homography
from PIL import Image


def get_dataloader(data_dir, phase, obs_len, pred_len, batch_size):
    r"""Get dataloader for a specific phase

    Args:
        data_dir (str): path to the dataset directory
        phase (str): phase of the data, one of 'train', 'val', 'test'
        obs_len (int): length of observed trajectory
        pred_len (int): length of predicted trajectory
        batch_size (int): batch size

    Returns:
        loader_phase (torch.utils.data.DataLoader): dataloader for the specific phase
    """

    assert phase in ['train', 'val', 'test']

    data_set = data_dir + '/' + phase + '/'
    shuffle = True if phase == 'train' else False
    drop_last = True if phase == 'train' else False

    dataset_phase = TrajectoryDataset(data_set, obs_len=obs_len, pred_len=pred_len)
    sampler_phase = None
    if batch_size > 1:
        sampler_phase = TrajBatchSampler(dataset_phase, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    loader_phase = DataLoader(dataset_phase, collate_fn=traj_collate_fn, batch_sampler=sampler_phase, pin_memory=True)
    return loader_phase


def traj_collate_fn(data):
    r"""Collate function for the dataloader

    Args:
        data (list): list of tuples of (obs_seq, pred_seq, non_linear_ped, loss_mask, seq_start_end, scene_id)

    Returns:
        obs_seq_list (torch.Tensor): (num_ped, obs_len, 2)
        pred_seq_list (torch.Tensor): (num_ped, pred_len, 2)
        non_linear_ped_list (torch.Tensor): (num_ped,)
        loss_mask_list (torch.Tensor): (num_ped, obs_len + pred_len)
        scene_mask (torch.Tensor): (num_ped, num_ped)
        seq_start_end (torch.Tensor): (num_ped, 2)
        scene_id
    """

    data_collated = {}
    for k in data[0].keys():
        data_collated[k] = [d[k] for d in data]

    _len = [len(seq) for seq in data_collated["obs_traj"]]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    seq_start_end = torch.LongTensor(seq_start_end)
    scene_mask = torch.zeros(sum(_len), sum(_len), dtype=torch.bool)
    for idx, (start, end) in enumerate(seq_start_end):
        scene_mask[start:end, start:end] = 1

    data_collated["obs_traj"] = torch.cat(data_collated["obs_traj"], dim=0)
    data_collated["pred_traj"] = torch.cat(data_collated["pred_traj"], dim=0)
    data_collated["non_linear_ped"] = torch.cat(data_collated["non_linear_ped"], dim=0)
    data_collated["loss_mask"] = torch.cat(data_collated["loss_mask"], dim=0)
    data_collated["scene_mask"] = scene_mask
    data_collated["seq_start_end"] = seq_start_end
    data_collated["frame"] = torch.cat(data_collated["frame"], dim=0)
    data_collated["scene_id"] = np.concatenate(data_collated["scene_id"], axis=0)

    return data_collated


class TrajBatchSampler(Sampler):
    r"""Samples batched elements by yielding a mini-batch of indices.
    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, batch_size=64, shuffle=False, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        assert len(self.data_source) == len(self.data_source.num_peds_in_seq)

        if self.shuffle:
            if self.generator is None:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            else:
                generator = self.generator
            indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        else:
            indices = list(range(len(self.data_source)))
        num_peds_indices = self.data_source.num_peds_in_seq[indices]

        batch = []
        total_num_peds = 0
        for idx, num_peds in zip(indices, num_peds_indices):
            batch.append(idx)
            total_num_peds += num_peds
            if total_num_peds >= self.batch_size:
                yield batch
                batch = []
                total_num_peds = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Approximated number of batches.
        # The order of trajectories can be shuffled, so this number can vary from run to run.
        if self.drop_last:
            return sum(self.data_source.num_peds_in_seq) // self.batch_size
        else:
            return (sum(self.data_source.num_peds_in_seq) + self.batch_size - 1) // self.batch_size


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non-linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.02, min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        non_linear_ped = []
        frame_list = []
        scene_id = []
        self.homography = {}
        self.scene_img = {}
        self.scene_map = {}
        self.scene_desc = {}
        scene_img_map = {'biwi_eth': 'seq_eth', 'biwi_hotel': 'seq_hotel',
                         'students001': 'students003', 'students003': 'students003', 'uni_examples': 'students003',
                         'crowds_zara01': 'crowds_zara01', 'crowds_zara02': 'crowds_zara02', 'crowds_zara03': 'crowds_zara02'}

        for path in all_files:
            # Load image
            parent_dir, scene_name = os.path.split(path)
            parent_dir, phase = os.path.split(parent_dir)
            parent_dir, dataset_name = os.path.split(parent_dir)
            scene_name, _ = os.path.splitext(scene_name)
            scene_name = scene_name.replace('_' + phase, '')

            try:
                self.scene_img[scene_name] = Image.open(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_reference.png"))
                self.scene_map[scene_name] = np.array(Image.open(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_oracle.png")))

                # check caption file exist
                if os.path.exists(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_caption.txt")):
                    with open(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_caption.txt"), "r") as f:
                        self.scene_desc[scene_name] = f.read()
                else:
                    self.scene_desc[scene_name] = ""
            except:
                self.scene_img[scene_name] = None
                self.scene_map[scene_name] = None
                self.scene_desc[scene_name] = ""

            # Load homography matrix
            if dataset_name in ["eth", "hotel", "univ", "zara1", "zara2", "rawall"]:
                homography_file = os.path.join(parent_dir, "homography", scene_name + "_H.txt")
                self.homography[scene_name] = np.loadtxt(homography_file)

            # Load data
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    frame_list.extend([frames[idx]] * num_peds_considered)
                    scene_id.extend([scene_name] * num_peds_considered)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.num_peds_in_seq = np.array(num_peds_in_seq)
        self.frame_list = np.array(frame_list, dtype=np.int32)
        self.scene_id = np.array(scene_id)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float).permute(0, 2, 1)  # NTC
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float).permute(0, 2, 1)  # NTC
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float).gt(0.5)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float).gt(0.5)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.frame_list = torch.from_numpy(self.frame_list).type(torch.long)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = {"obs_traj": self.obs_traj[start:end],
               "pred_traj": self.pred_traj[start:end],
               "non_linear_ped": self.non_linear_ped[start:end],
               "loss_mask": self.loss_mask[start:end],
               "scene_mask": None,
               "seq_start_end": [[0, end - start]],
               "frame": self.frame_list[start:end],
               "scene_id": self.scene_id[start:end]}
        return out
