import torch
import os
import torch.distributed as dist
import numpy as np


class ShardDataLoader:
    # TODO I should really just store my train and val datasets in two different memmaps rather than a bunch of numpy arrays and pin memory when moving to cuda, but doesn't matter for now
    # I'm also not going to bother resuming training runs with proper rng seed for times for now
    def __init__(self, shard_dir, batch_size, seq_len, rng_seed=21):

        self.batch_size = batch_size
        self.seq_len = seq_len

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0    

        shards = sorted(os.listdir(shard_dir))
        self.shards = [os.path.join(shard_dir, shard_path) for shard_path in shards]
        assert len(self.shards) > 0, f"{len(shards)=}, {len(self.shards)=}, {shard_dir=}"

        torch.manual_seed(rng_seed + self.rank) 

        self.reset()

    def load_tokens(self):
        tokens = np.load(self.shards[self.shard_index])
        if not tokens.flags["C_CONTIGUOUS"]:
            raise ValueError(f"numpy shard {self.shards[self.shard_index]} was not saved in contiguous memory")
        self.tokens = torch.from_numpy(tokens)
    
    def reset(self):
        self.shard_index = 0
        self.load_tokens()
        self.current_position = self.rank * self.batch_size * self.seq_len

    def next_batch(self):
        if self.tokens is None:
            print(f"{self.current_position=}, {self.shard_index=}, {self.batch_size=}, {self.seq_len=}, {self.shards=}")
        x = self.tokens[self.current_position : self.current_position + self.batch_size * self.seq_len].view(self.batch_size, self.seq_len)
        t = torch.rand(self.batch_size)

        self.current_position += self.batch_size * self.seq_len * self.world_size
        # if loading the next batch would be out of bounds, advance to the next shard
        if self.current_position + self.batch_size * self.seq_len * self.world_size >= len(self.tokens):
            self.shard_index = (self.shard_index + 1) % len(self.shards)
            self.load_tokens()
            self.current_position = self.rank * self.batch_size * self.seq_len
        return x, t

    def get_state_dict(self):
        if self.rank == 0:
            return {"current_position": self.current_position, "shard_index": self.shard_index}
        raise RuntimeError(f"get_state_dict can only be called on rank 0, but {self.rank=}")

    def load_state_dict(self, state_dict):
        """
        `state_dict` is a dictionary that corresponds to rank 0. Since the values in it are from rank 0, we need to
        adjust the current position and potentially the shard index for other ranks
        """
        if state_dict is not None:
            self.current_position = state_dict["current_position"] + self.rank * self.batch_size * self.seq_len
            self.shard_index = state_dict["shard_index"] # rank 0's shard index, we might need to change it for other ranks
            self.load_tokens()
            # if the loaded current position (based on rank 0's current position) is out of bounds, then move to the next shard
            if self.current_position >= len(self.tokens):
                if self.rank == 0:
                    raise ValueError(f"Rank 0 should not have an incorrect shard index, check the validity of your checkpoint!")
                self.shard_index = (self.shard_index + 1) % len(self.shards)
                self.load_tokens()
                self.current_position = self.rank * self.batch_size * self.seq_len
