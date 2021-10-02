from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING

import numpy as np
import torch
from shade.alias import Action
from shade.alias import ActionBatch
from shade.alias import DoneBatch
from shade.alias import InfoBatch
from shade.alias import ObsBatch
from shade.alias import RewardBatch
from shade.alias import Space
from shade.alias import TensorDict
from shade.module import ManagerModule
from shade.registry import register
from shade.util import dlist_to_gend
from torch import multiprocessing as mp

if TYPE_CHECKING:
    from shade.module import EnvModule


def _shared_memory_dict(
    space: Space, dtypes: Dict[str, torch.dtype], batch_sz: int
) -> TensorDict:
    return {
        key: torch.zeros(
            (batch_sz, *shp),
            dtype=dtypes[key],
        ).share_memory_()
        for key, shp in space.items()
    }


# TODO https://github.com/heronsystems/shade/issues/8
@register
class SharedMemoryManager(ManagerModule):
    def __init__(
        self,
        env_cls: Type[EnvModule],
        batch_sz: Optional[int] = None,
        start_seed: Optional[int] = None,
    ):
        global_cfg = self.global_cfg()
        self._batch_sz = batch_sz or global_cfg.batch_sz
        self._start_seed = start_seed or global_cfg.seed
        tmp_env = env_cls(0)
        tmp_env.close()
        self._obs_sm = _shared_memory_dict(
            tmp_env.observation_space_cpu,
            tmp_env.observation_dtypes_cpu,
            self._batch_sz,
        )
        self._reward_sm = torch.zeros(self._batch_sz).share_memory_()
        self._done_sm = torch.zeros(
            self._batch_sz, dtype=torch.bool
        ).share_memory_()
        self._info_sm = _shared_memory_dict(
            tmp_env.info_space, tmp_env.info_dtype, self._batch_sz
        )
        self._procs = []
        self._cxns = []
        for batch_ix in range(self._batch_sz):
            seed = self._start_seed + batch_ix
            parent_cxn, child_cxn = mp.Pipe()
            proc = mp.Process(
                target=worker,
                args=(
                    batch_ix,
                    child_cxn,
                    env_cls,
                    seed,
                    self._obs_sm,
                    self._reward_sm,
                    self._done_sm,
                ),
                daemon=True,
            )
            proc.start()
            self._cxns.append(parent_cxn)
            self._procs.append(proc)
        self._act_sm = self._get_act_sm()

    def step(
        self, action_batch: ActionBatch
    ) -> Tuple[ObsBatch, RewardBatch, DoneBatch, InfoBatch]:
        action_gen = dlist_to_gend(action_batch)
        for batch_ix in range(self._batch_sz):
            action = next(action_gen)
            for act_key, act_tensor in action.items():
                self._act_sm[batch_ix][act_key].copy_(act_tensor)
            self._cxns[batch_ix].send("step")
        infos = []
        for cxn in self._cxns:
            infos.append(cxn.recv())
        return (self._obs_sm, self._reward_sm, self._done_sm, infos)

    def reset(self) -> ObsBatch:
        for cxn in self._cxns:
            cxn.send("reset")
        for cxn in self._cxns:
            cxn.recv()
        return self._obs_sm

    def close(self) -> None:
        for c in self._cxns:
            c.send("close")
        for p in self._procs:
            p.join()

    def _get_act_sm(self) -> List[Action]:
        for cxn in self._cxns:
            cxn.send("get_action_sm")
        return [cxn.recv() for cxn in self._cxns]


def worker(
    batch_ix: int,
    cxn: Connection,
    env_cls: Type[EnvModule],
    seed: int,
    obs_sm: TensorDict,
    reward_sm: torch.Tensor,
    done_sm: torch.BoolTensor,
):
    env = env_cls(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    action_sm = {
        act_key: torch.zeros(
            act_shp, dtype=env.action_dtype[act_key]
        ).share_memory_()
        for act_key, act_shp in env.action_space.items()
    }
    running = True
    while running:
        try:
            msg = cxn.recv()
            if msg == "step":
                obs, reward, done, info = env.step(action_sm)
                if done:
                    obs = env.reset()
                for k, v in obs.items():
                    obs_sm[k][batch_ix] = torch.tensor(v)
                reward_sm[batch_ix] = torch.tensor(reward)
                done_sm[batch_ix] = done
                cxn.send(info)
            elif msg == "reset":
                obs = env.reset()
                for k, v in obs.items():
                    obs_sm[k][batch_ix] = torch.tensor(v)
                cxn.send(True)
            elif msg == "close":
                env.close()
                running = False
            elif msg == "get_action_sm":
                cxn.send(action_sm)
            else:
                raise Exception(f"Unexpected message: {msg}")
        except KeyboardInterrupt:
            env.close()
            running = False


if __name__ == "__main__":
    from shade.env import AtariEnv
    from shade.config import load_cfg

    cfg = load_cfg()
    mgr = SharedMemoryManager(AtariEnv, batch_sz=2, start_seed=0)
    mgr.reset()
    action_batch = {"action": torch.ones(cfg.batch_sz).long()}
    obs, rew, done, info = mgr.step(action_batch)
    mgr.close()
