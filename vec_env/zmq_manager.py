from __future__ import annotations

from multiprocessing.connection import Connection
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING

import msgpack
import torch
import zmq
from shade.alias import Action
from shade.alias import ActionBatch
from shade.alias import DoneBatch
from shade.alias import InfoBatch
from shade.alias import ObsBatch
from shade.alias import RewardBatch
from shade.alias import TensorDict
from shade.module.mgr_module import ManagerModule
from shade.registry import register
from shade.util import dlist_to_gend
from torch import multiprocessing as mp

if TYPE_CHECKING:
    from shade.module import EnvModule


@register
class ZMQManager(ManagerModule):
    def __init__(
        self,
        env_cls: Type[EnvModule],
        batch_sz: Optional[int] = None,
        start_seed: Optional[int] = None,
    ):
        cfg = self.global_cfg()
        self._env_cls = env_cls
        self._batch_sz = batch_sz or cfg.batch_sz
        self._start_seed = start_seed or cfg.seed
        tmp_env = env_cls(0)
        tmp_env.close()
        self._obs_sm = {
            obs_key: torch.zeros(
                (self._batch_sz, *obs_shp),
                dtype=tmp_env.observation_dtypes_cpu[obs_key],
            ).share_memory_()
            for obs_key, obs_shp in tmp_env.observation_space_cpu.items()
        }
        self._reward_sm = torch.zeros(self._batch_sz).share_memory_()
        self._done_sm = torch.zeros(
            self._batch_sz, dtype=torch.bool
        ).share_memory_()
        self._zmq_context = zmq.Context()
        self._procs = []
        self._cxns = []
        self._sockets = []
        for batch_ix in range(self._batch_sz):
            seed = self._start_seed + batch_ix
            parent_cxn, child_cxn = mp.Pipe()
            socket = self._zmq_context.socket(zmq.PAIR)
            port = socket.bind_to_random_port("tcp://*")
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
                    port,
                ),
                daemon=True,
            )
            proc.start()
            self._cxns.append(parent_cxn)
            self._procs.append(proc)
            self._sockets.append(socket)
        self._act_sm = self._get_act_sm()
        for cxn in self._cxns:
            cxn.send("switch_zmq")

    def step(
        self, action_batch: ActionBatch
    ) -> Tuple[ObsBatch, RewardBatch, DoneBatch, InfoBatch]:
        action_gen = dlist_to_gend(action_batch)
        for batch_ix in range(self._batch_sz):
            action = next(action_gen)
            for act_key, act_tensor in action.items():
                self._act_sm[batch_ix][act_key].copy_(act_tensor)
            self._sockets[batch_ix].send(
                msgpack.dumps("step"), zmq.NOBLOCK, copy=False
            )
        infos = []
        for s in self._sockets:
            infos.append(msgpack.loads(s.recv()))
        return (
            self._obs_sm,
            self._reward_sm,
            self._done_sm,
            infos,
        )

    def reset(self) -> ObsBatch:
        for s in self._sockets:
            s.send(msgpack.dumps("reset"), zmq.NOBLOCK, copy=False)
        for s in self._sockets:
            s.recv()
        return self._obs_sm

    def close(self) -> None:
        for s in self._sockets:
            s.send(msgpack.dumps("close"), zmq.NOBLOCK, copy=False)
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
    zmq_port: int,
):
    env = env_cls(seed)
    action_sm = {
        act_key: torch.zeros(
            act_shp, dtype=env.action_dtype[act_key]
        ).share_memory_()
        for act_key, act_shp in env.action_space.items()
    }
    pipe = True
    while pipe:
        msg = cxn.recv()
        if msg == "get_action_sm":
            cxn.send(action_sm)
        elif msg == "switch_zmq":
            cxn.close()
            pipe = False
        else:
            raise Exception(f"Unexpected message: {msg}")

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(f"tcp://localhost:{zmq_port}")

    zmq_loop = True
    while zmq_loop:
        msg = msgpack.loads(socket.recv())
        if msg == "step":
            obs, reward, done, info = env.step(action_sm)
            if done:
                obs = env.reset()
            for k, v in obs.items():
                obs_sm[k][batch_ix] = v
            reward_sm[batch_ix] = reward
            done_sm[batch_ix] = done
            socket.send(msgpack.dumps(info), zmq.NOBLOCK, copy=False)
        elif msg == "reset":
            obs = env.reset()
            for k, v in obs.items():
                obs_sm[k][batch_ix] = v
            socket.send(msgpack.dumps(True), zmq.NOBLOCK, copy=False)
        elif msg == "close":
            env.close()
            zmq_loop = False


if __name__ == "__main__":
    from shade.env import AtariEnv
    from shade.config import load_cfg

    cfg = load_cfg()
    mgr = ZMQManager(AtariEnv, batch_sz=2, start_seed=0)
    mgr.reset()
    action_batch = {"action": torch.ones(cfg.batch_sz).long()}
    obs, rew, done, info = mgr.step(action_batch)
    mgr.close()
