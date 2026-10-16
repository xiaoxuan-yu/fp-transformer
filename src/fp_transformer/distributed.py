from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    distributed_enabled: bool
    launcher: str
    world_size: int
    world_rank: int
    local_size: int
    local_rank: int
    global_rank: int
    num_nodes: int
    total_devices: int
    coordinator_address: str | None


def _get_env_first(keys: list[str]) -> str | None:
    for key in keys:
        v = os.environ.get(key)
        if v:
            return v
    return None


def _parse_int_like(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    # Slurm may expose values like "8(x2)"
    m = re.match(r"^\s*(\d+)", value)
    if m:
        return int(m.group(1))
    return None


def _get_first_env_int(keys: list[str]) -> int | None:
    for key in keys:
        v = _parse_int_like(os.environ.get(key))
        if v is not None:
            return v
    return None


def _infer_launcher() -> str:
    if os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("OMPI_COMM_WORLD_SIZE"):
        return "ompi"
    if (
        os.environ.get("SLURM_PROCID")
        or os.environ.get("SLURM_NTASKS")
        or os.environ.get("PMI_RANK")
        or os.environ.get("PMI_SIZE")
        or os.environ.get("PMIX_RANK")
    ):
        return "slurm"
    return "single"


def _expand_slurm_nodelist_first(nodelist: str | None) -> str | None:
    if not nodelist:
        return None
    # Handles simple forms like "node001" and "node[001-004,010]"
    if "[" not in nodelist:
        return nodelist.split(",")[0]
    m = re.match(r"^([^\[]+)\[([^\]]+)\](.*)$", nodelist)
    if not m:
        return None
    prefix, body, suffix = m.groups()
    first_group = body.split(",")[0]
    first_token = first_group.split("-")[0]
    return f"{prefix}{first_token}{suffix}"


def _infer_coordinator_host(
    launcher: str, world_size: int, num_nodes_hint: int | None
) -> str | None:
    host = _get_env_first(["MASTER_ADDR"])
    if host:
        return host
    coord_addr = _get_env_first(["JAX_COORDINATOR_ADDRESS"])
    if coord_addr and ":" in coord_addr:
        return coord_addr.rsplit(":", 1)[0]
    if coord_addr:
        return coord_addr
    if launcher == "slurm":
        host = _get_env_first(["SLURM_LAUNCH_NODE_IPADDR"])
        if host:
            return host
        host = _expand_slurm_nodelist_first(os.environ.get("SLURM_NODELIST"))
        if host:
            return host
    if world_size <= 1 or (num_nodes_hint is not None and num_nodes_hint <= 1):
        return "127.0.0.1"
    return None


def _infer_coordinator_port() -> str | None:
    port = _get_env_first(["MASTER_PORT", "JAX_COORDINATOR_PORT"])
    if port:
        return port
    # Stable default for scripted runs when not provided explicitly.
    return "12355"


def init_jax_distributed(
    jax_dist,
    jax_module,
    *,
    cluster_detection_method: str | None = "auto",
) -> DistributedContext:
    launcher = _infer_launcher()

    world_size = _get_first_env_int(
        ["OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS", "PMI_SIZE"]
    )
    world_rank = _get_first_env_int(
        ["OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "PMI_RANK", "PMIX_RANK"]
    )
    local_rank = _get_first_env_int(
        ["OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "PMI_LOCAL_RANK", "LOCAL_RANK"]
    )
    local_size = _get_first_env_int(
        ["OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE", "LOCAL_WORLD_SIZE"]
    )

    if world_size is None:
        world_size = 1
    if world_rank is None:
        world_rank = 0
    if local_rank is None:
        local_rank = 0
    if local_size is None or local_size < 1:
        local_size = int(jax_module.local_device_count())
        if local_size < 1:
            local_size = 1

    num_nodes_hint = max(1, int(world_size) // max(1, int(local_size)))
    master_addr = _infer_coordinator_host(launcher, int(world_size), num_nodes_hint)
    master_port = _infer_coordinator_port()
    coordinator_address = (
        f"{master_addr}:{master_port}"
        if master_addr is not None and master_port
        else None
    )

    distributed_enabled = (
        int(world_size) > 1
        and coordinator_address is not None
        and int(world_rank) >= 0
        and int(local_rank) >= 0
    )

    if int(world_size) > 1 and not distributed_enabled:
        raise RuntimeError(
            "Detected multi-process launcher environment but failed to infer JAX "
            "distributed coordinator address. Set MASTER_ADDR and MASTER_PORT."
        )

    if distributed_enabled:
        init_kwargs = dict(
            coordinator_address=coordinator_address,
            num_processes=int(world_size),
            process_id=int(world_rank),
            local_device_ids=[int(local_rank)],
        )
        # Ranks and coordinator are taken from the launcher environment; a
        # launch script may still request e.g. "mpi4py" explicitly.
        if cluster_detection_method not in (None, "auto"):
            init_kwargs["cluster_detection_method"] = cluster_detection_method
        jax_dist.initialize(**init_kwargs)

    global_rank = int(jax_module.process_index())
    total_devices = int(jax_module.device_count())
    local_dev_count = int(jax_module.local_device_count())
    if local_dev_count > 0:
        local_size = local_dev_count

    if local_size < 1:
        local_size = 1
    num_nodes = max(1, int(world_size) // int(local_size))

    return DistributedContext(
        distributed_enabled=bool(distributed_enabled),
        launcher=launcher,
        world_size=int(world_size),
        world_rank=int(world_rank),
        local_size=int(local_size),
        local_rank=int(local_rank),
        global_rank=int(global_rank),
        num_nodes=int(num_nodes),
        total_devices=int(total_devices),
        coordinator_address=coordinator_address,
    )


def create_process_mesh(ctx: DistributedContext, *, axis_names=("node", "device")):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    if ctx.total_devices != ctx.num_nodes * ctx.local_size:
        raise ValueError(
            f"Total devices ({ctx.total_devices}) != nodes ({ctx.num_nodes}) * "
            f"local_devices ({ctx.local_size})."
        )
    device_mesh = mesh_utils.create_device_mesh((ctx.num_nodes, ctx.local_size))
    return Mesh(device_mesh, axis_names=axis_names)


def gather_across_hosts(array, *, ctx: DistributedContext):
    from jax.experimental import multihost_utils

    if not ctx.distributed_enabled:
        return array
    gathered = multihost_utils.process_allgather(array)
    if tuple(getattr(gathered, "shape", ())) == tuple(getattr(array, "shape", ())):
        return gathered
    if gathered.ndim == array.ndim + 1 and gathered.shape[0] == ctx.world_size:
        return gathered.reshape((-1,) + gathered.shape[2:])
    if gathered.ndim == array.ndim + 1 and gathered.shape[1:] == array.shape:
        return gathered[0]
    return gathered


def sync_hosts(tag: str, *, ctx: DistributedContext) -> None:
    from jax.experimental import multihost_utils

    if not ctx.distributed_enabled:
        return
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def safe_shutdown() -> None:
    from jax import distributed as jax_dist

    try:
        jax_dist.shutdown()
    except Exception:
        pass
