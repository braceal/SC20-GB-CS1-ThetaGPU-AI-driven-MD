import os
import yaml
import argparse
import simtk.unit as u
from deepdrivemd.sim.openmm_sim import run_simulation
from deepdrivemd import config_logging

import logging

logger = logging.getLogger(__name__)


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="YAML config file", required=True)
    path = parser.parse_args().config
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


def build_simulation_params(cfg: dict) -> dict:
    return dict(
        reference_pdb_file=cfg["reference_pdb_file"],
        omm_dir_prefix=cfg["omm_dir_prefix"],
        local_run_dir=cfg["local_run_dir"],
        gpu_index=0,
        sim_type=cfg["sim_type"],
        report_interval_ps=float(cfg["report_interval_ps"]) * u.picoseconds,
        frames_per_h5=cfg["frames_per_h5"],
        sim_time=float(cfg["simulation_length_ns"]) * u.nanoseconds,
        dt_ps=float(cfg["dt_ps"]) * u.picoseconds,
        temperature_kelvin=float(cfg["temperature_kelvin"]) * u.kelvin,
        h5_scp_path=cfg["h5_scp_path"],
        result_dir=cfg["result_dir"],
        input_dir=cfg["input_dir"],
        initial_configs_dir=cfg["initial_configs_dir"],
        wrap=cfg["wrap"],
    )


if __name__ == "__main__":
    cfg = get_config()
    log_fname = os.path.join(cfg["result_dir"], cfg["omm_dir_prefix"] + ".log")
    config_logging(filename=log_fname, **cfg["logging"])
    logger = logging.getLogger(__name__)
    simulation_kwargs = build_simulation_params(cfg)
    kw_str = ", ".join(f"{k}={v}" for k, v in simulation_kwargs.items())
    logger.debug(f"run_simulation({kw_str})")
    run_simulation(**simulation_kwargs)
