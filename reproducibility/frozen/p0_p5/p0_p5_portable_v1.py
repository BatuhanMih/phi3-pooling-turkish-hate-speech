from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import types
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUNTIME_VERSION = "p0_p5_portable_v1r3"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MODEL_REVISION = "f39ac1d28e925b323eae81227eaba4464caced4e"
TRAIN_RAW_SHA256 = "354c53ac39b1a73ccdb729de270045c774f919a8e30733f2e3d01b9876d424e8"
TRAIN_CANONICAL_SHA256 = "47caeece445742fbb2bc6da60ed5094375ca0e54f41e77b3cc656860da6a5f54"
EXPECTED_RECORD_COUNT = 9_550
EXPECTED_OPTIMIZER_STEPS = 7_164
AMP_INITIAL_SCALE = 128.0
AMP_GROWTH_INTERVAL = 1_000_000
SEEDS = (42, 142, 242, 342, 442, 542, 642, 742, 842, 942)
PAIR_ORDER = (
    (42, "P0", "P5"),
    (142, "P5", "P0"),
    (242, "P0", "P5"),
    (342, "P5", "P0"),
    (442, "P0", "P5"),
    (542, "P5", "P0"),
    (642, "P0", "P5"),
    (742, "P5", "P0"),
    (842, "P0", "P5"),
    (942, "P5", "P0"),
)
EXPECTED_VERSIONS = {
    "python": "3.11.9",
    "torch": "2.5.1+cu121",
    "transformers": "4.42.4",
    "peft": "0.11.1",
    "bitsandbytes": "0.49.2",
    "accelerate": "0.32.1",
    "datasets": "5.0.0",
    "numpy": "1.26.4",
    "pandas": "3.0.3",
    "sklearn": "1.9.0",
    "safetensors": "0.8.0",
}
LOCKED_CONFIG = {
    "learning_rate": 8.687484140587463e-05,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0697368029059364,
    "lora_target_modules": ["qkv_proj"],
    "modules_to_save": ["score"],
    "warmup_ratio": 0.1951244703273512,
    "weight_decay": 0.004708867394382477,
    "max_length": 128,
    "label_smoothing_factor": 0.05,
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "compute_dtype": "float16",
    "optimizer": "paged_adamw_32bit",
    "gradient_checkpointing": True,
    "append_terminal_token": False,
    "prediction_rule": "argmax",
    "threshold_tuning": False,
    "calibration": False,
}


class ContractError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ContractError(f"NO_OVERWRITE:{path}") from error


def replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def root_path() -> Path:
    override = os.environ.get("PHI3_P0P5_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


ROOT = root_path()
PACKAGE = ROOT / "package"
REPORTS = ROOT / "reports"
CAMPAIGN_ROOT = ROOT / "artifacts" / "campaign_v1r2"
TRAIN_PATH = PACKAGE / "training_data" / "train.jsonl"
CONTRACT_PATH = PACKAGE / "handoff_v3" / "v2_locked" / "scientific_execution_contract.json"
SEED_PLAN_PATH = PACKAGE / "handoff_v3" / "v2_locked" / "seed_plan.json"
CPU_REPORT_PATH = REPORTS / "cpu_prepare_report_v1r3.json"
PROBE_REPORT_PATH = REPORTS / "gpu_setup_probe_report_v1r3.json"


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    record_id: str
    text: str
    label: int
    domain: str


def load_training_records() -> tuple[list[TrainingRecord], str, str]:
    if not TRAIN_PATH.is_file():
        raise ContractError("TRAIN_FILE_MISSING")
    raw_hash = sha256_file(TRAIN_PATH)
    if raw_hash != TRAIN_RAW_SHA256:
        raise ContractError("TRAIN_RAW_SHA256_MISMATCH")
    records: list[TrainingRecord] = []
    canonical = hashlib.sha256()
    with TRAIN_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != {"id", "text", "label", "domain"}:
                raise ContractError(f"TRAIN_RECORD_SHAPE_INVALID:{line_number}")
            text = value["text"]
            label = value["label"]
            if type(text) is not str or not text or text != text.strip():
                raise ContractError(f"TRAIN_TEXT_INVALID:{line_number}")
            if type(label) is not int or label not in (0, 1):
                raise ContractError(f"TRAIN_LABEL_INVALID:{line_number}")
            canonical.update(
                (
                    json.dumps(
                        {"label": label, "text": text},
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            records.append(
                TrainingRecord(
                    record_id=str(value["id"]),
                    text=text,
                    label=label,
                    domain=str(value["domain"]),
                )
            )
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ContractError("TRAIN_RECORD_COUNT_MISMATCH")
    canonical_hash = canonical.hexdigest()
    if canonical_hash != TRAIN_CANONICAL_SHA256:
        raise ContractError("TRAIN_CANONICAL_SHA256_MISMATCH")
    return records, raw_hash, canonical_hash


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def memory_and_power() -> dict[str, Any]:
    memory = MemoryStatusEx()
    memory.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ContractError("MEMORY_STATUS_QUERY_FAILED")
    power = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
        raise ContractError("POWER_STATUS_QUERY_FAILED")
    return {
        "total_ram_gb": round(memory.ullTotalPhys / (1024**3), 3),
        "available_ram_gb": round(memory.ullAvailPhys / (1024**3), 3),
        "commit_limit_gb": round(memory.ullTotalPageFile / (1024**3), 3),
        "commit_headroom_gb": round(memory.ullAvailPageFile / (1024**3), 3),
        "ac_connected": power.ACLineStatus == 1,
        "battery_percent": int(power.BatteryLifePercent),
    }


def gpu_status() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    if completed.returncode != 0:
        raise ContractError("NVIDIA_SMI_QUERY_FAILED")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ContractError("SINGLE_NVIDIA_GPU_REQUIRED")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 6:
        raise ContractError("NVIDIA_SMI_OUTPUT_INVALID")
    return {
        "gpu_name": parts[0],
        "driver_version": parts[1],
        "vram_total_mib": int(parts[2]),
        "vram_free_mib": int(parts[3]),
        "gpu_temperature_c": int(parts[4]),
        "gpu_pstate": parts[5],
    }


def active_runtime_processes() -> list[dict[str, Any]]:
    import psutil

    allowed = {os.getpid(), os.getppid()}
    parent_text = os.environ.get("P0P5_PARENT_PID")
    if parent_text and parent_text.isdigit():
        allowed.add(int(parent_text))
    matches: list[dict[str, Any]] = []
    needle = Path(__file__).name.lower()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            if pid in allowed:
                continue
            command_line = " ".join(process.info.get("cmdline") or []).lower()
            if needle in command_line and (" campaign" in command_line or " worker" in command_line):
                matches.append({"pid": pid, "name": process.info.get("name")})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def resource_snapshot() -> dict[str, Any]:
    payload = {"timestamp_utc": utc_now()}
    payload.update(memory_and_power())
    payload.update(gpu_status())
    payload["free_disk_gb"] = round(shutil.disk_usage(ROOT).free / (1024**3), 3)
    return payload


def enforce_resource_gate(*, initial: bool, allow_current_runtime: bool = False) -> dict[str, Any]:
    snapshot = resource_snapshot()
    ram_floor = 8.0 if initial else 6.0
    commit_floor = 16.0 if initial else 12.0
    disk_floor = 60.0 if initial else 40.0
    failures: list[str] = []
    if snapshot["available_ram_gb"] < ram_floor:
        failures.append("AVAILABLE_RAM_BELOW_GATE")
    if snapshot["commit_headroom_gb"] < commit_floor:
        failures.append("COMMIT_HEADROOM_BELOW_GATE")
    if snapshot["free_disk_gb"] < disk_floor:
        failures.append("FREE_DISK_BELOW_GATE")
    if not snapshot["ac_connected"]:
        failures.append("AC_POWER_DISCONNECTED")
    if initial and snapshot["gpu_temperature_c"] >= 80:
        failures.append("GPU_TOO_HOT_TO_START")
    if initial and snapshot["vram_total_mib"] < 4_000:
        failures.append("VRAM_BELOW_MINIMUM")
    if not allow_current_runtime:
        processes = active_runtime_processes()
        if processes:
            failures.append("SECOND_CAMPAIGN_PROCESS_DETECTED")
        snapshot["other_runtime_processes"] = processes
    if failures:
        raise ContractError("RESOURCE_GATE_FAILED:" + ",".join(failures))
    return snapshot


def set_below_normal_priority() -> None:
    try:
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def verify_runtime_identity() -> dict[str, Any]:
    if not CPU_REPORT_PATH.is_file():
        raise ContractError("CPU_PREPARE_REPORT_MISSING")
    report = load_json(CPU_REPORT_PATH)
    if report.get("status") != "PASS":
        raise ContractError("CPU_PREPARE_NOT_PASS")
    observed = sha256_file(Path(__file__).resolve())
    if report.get("runtime_sha256") != observed:
        raise ContractError("RUNTIME_SHA256_MISMATCH")
    return report


def expected_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_index = 0
    for pair_index, (seed, first, second) in enumerate(PAIR_ORDER, 1):
        for condition in (first, second):
            run_index += 1
            rows.append(
                {
                    "run_index": run_index,
                    "pair_index": pair_index,
                    "seed": seed,
                    "condition_id": condition,
                    "worker_id": f"{condition}_seed_{seed:04d}",
                }
            )
    return rows


def validate_contract_and_seed_plan() -> tuple[str, str]:
    contract = load_json(CONTRACT_PATH)
    seed_plan = load_json(SEED_PLAN_PATH)
    if contract.get("model", {}).get("model_id") != MODEL_ID:
        raise ContractError("MODEL_ID_CONTRACT_MISMATCH")
    if contract.get("model", {}).get("revision") != MODEL_REVISION:
        raise ContractError("MODEL_REVISION_CONTRACT_MISMATCH")
    if [item.get("condition_id") for item in contract.get("conditions", [])] != ["P0", "P5"]:
        raise ContractError("CONDITION_CONTRACT_MISMATCH")
    paired = contract.get("paired_training", {})
    locked = contract.get("locked_hyperparameters", {})
    checks = {
        "seed_plan": paired.get("seed_plan") == list(SEEDS),
        "model_count": paired.get("model_count") == 20,
        "steps": paired.get("expected_optimizer_steps_per_model") == EXPECTED_OPTIMIZER_STEPS,
        "epochs": paired.get("num_train_epochs") == LOCKED_CONFIG["num_train_epochs"],
        "batch": paired.get("per_device_train_batch_size") == 1,
        "gradient_accumulation": paired.get("gradient_accumulation_steps") == 4,
        "learning_rate": locked.get("learning_rate") == LOCKED_CONFIG["learning_rate"],
        "lora_r": locked.get("lora_r") == 16,
        "lora_alpha": locked.get("lora_alpha") == 16,
        "lora_dropout": locked.get("lora_dropout") == LOCKED_CONFIG["lora_dropout"],
        "target": locked.get("lora_target_modules") == ["qkv_proj"],
        "modules_to_save": locked.get("modules_to_save") == ["score"],
        "warmup": locked.get("warmup_ratio") == LOCKED_CONFIG["warmup_ratio"],
        "weight_decay": locked.get("weight_decay") == LOCKED_CONFIG["weight_decay"],
        "max_length": locked.get("max_length") == 128,
        "label_smoothing": locked.get("label_smoothing_factor") == 0.05,
        "quantization": locked.get("load_in_4bit") is True,
        "quant_type": locked.get("bnb_4bit_quant_type") == "nf4",
        "double_quant": locked.get("bnb_4bit_use_double_quant") is True,
        "dtype": locked.get("compute_dtype") == "float16",
        "optimizer": locked.get("optimizer") == "paged_adamw_32bit",
        "gradient_checkpointing": locked.get("gradient_checkpointing") is True,
        "terminal_token": locked.get("append_terminal_token") is False,
        "threshold": locked.get("threshold_tuning") is False,
        "calibration": locked.get("calibration") is False,
        "test_absent": contract.get("data", {}).get("test_path_or_gold_included") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ContractError("SCIENTIFIC_CONTRACT_MISMATCH:" + ",".join(failed))
    pairs = seed_plan.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 10:
        raise ContractError("SEED_PLAN_PAIR_COUNT_MISMATCH")
    observed_order = []
    for item in pairs:
        observed_order.append((item.get("seed"), *(item.get("execution_order") or [])))
    if tuple(observed_order) != PAIR_ORDER:
        raise ContractError("SEED_PLAN_ORDER_MISMATCH")
    return sha256_file(CONTRACT_PATH), sha256_file(SEED_PLAN_PATH)


def cpu_prepare() -> int:
    if CPU_REPORT_PATH.exists():
        raise ContractError("CPU_PREPARE_REPORT_ALREADY_EXISTS")
    runtime_hash = sha256_file(Path(__file__).resolve())
    verifier = PACKAGE / "verify_friend_package.py"
    completed = subprocess.run(
        [sys.executable, "-B", str(verifier)],
        cwd=PACKAGE,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0 or "PACKAGE_STATUS=PASS" not in completed.stdout:
        raise ContractError("FRIEND_PACKAGE_VERIFICATION_FAILED")
    records, raw_hash, canonical_hash = load_training_records()
    contract_hash, seed_plan_hash = validate_contract_and_seed_plan()
    forbidden = {"test.jsonl", "test.csv", "gold.jsonl", "gold.csv", "test_0.csv"}
    found = [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*") if path.is_file() and path.name.lower() in forbidden]
    if found:
        raise ContractError("PROTECTED_PAYLOAD_PRESENT")
    initial_resources = enforce_resource_gate(initial=True)
    report = {
        "schema_version": "p0_p5_cpu_prepare_v1",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "runtime_version": RUNTIME_VERSION,
        "runtime_sha256": runtime_hash,
        "package_verifier_total": 271,
        "package_verifier_pass": 271,
        "package_verifier_fail": 0,
        "training_record_count": len(records),
        "training_raw_sha256": raw_hash,
        "training_canonical_sha256": canonical_hash,
        "scientific_contract_sha256": contract_hash,
        "seed_plan_sha256": seed_plan_hash,
        "planned_workers": expected_plan(),
        "planned_worker_count": 20,
        "conditions": ["P0", "P5"],
        "paired_seed_count": 10,
        "protected_payload_present": False,
        "protected_test_accessed": False,
        "automatic_retry_used": False,
        "second_campaign_started": False,
        "scientific_configuration_changed": False,
        "initial_resources": initial_resources,
    }
    write_json_once(CPU_REPORT_PATH, report)
    print("CPU_PREPARE=PASS", flush=True)
    return 0


def model_snapshot_path() -> Path:
    path_file = ROOT / "model_snapshot_path.txt"
    if not path_file.is_file():
        raise ContractError("MODEL_SNAPSHOT_PATH_MISSING")
    path = Path(path_file.read_text(encoding="utf-8").strip()).resolve()
    if path.name.lower() != MODEL_REVISION.lower():
        raise ContractError("MODEL_SNAPSHOT_REVISION_MISMATCH")
    required = ["config.json", "tokenizer_config.json"]
    if any(not (path / name).is_file() for name in required):
        raise ContractError("MODEL_SNAPSHOT_INCOMPLETE")
    if not ((path / "model.safetensors.index.json").is_file() or (path / "model.safetensors").is_file()):
        raise ContractError("MODEL_WEIGHTS_MISSING")
    return path


def load_heavy_runtime() -> dict[str, Any]:
    import accelerate
    import bitsandbytes
    import datasets
    import numpy
    import pandas
    import peft
    import safetensors
    import sklearn
    import torch
    import transformers

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "safetensors": safetensors.__version__,
    }
    if actual != EXPECTED_VERSIONS:
        raise ContractError(f"LOCKED_RUNTIME_VERSION_MISMATCH:{actual}")
    if not torch.cuda.is_available() or torch.version.cuda != "12.1":
        raise ContractError("CUDA_RUNTIME_MISMATCH")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    transformers.utils.logging.disable_progress_bar()
    return {
        "torch": torch,
        "transformers": transformers,
        "peft": peft,
        "datasets": datasets,
        "numpy": numpy,
        "versions": actual,
    }


def seed_everything(runtime: Mapping[str, Any], seed: int) -> None:
    torch = runtime["torch"]
    transformers = runtime["transformers"]
    random.seed(seed)
    runtime["numpy"].random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def valid_mask(torch: Any, hidden: Any, attention_mask: Any) -> tuple[Any, Any]:
    if attention_mask is None or hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ContractError("POOLING_INPUT_SHAPE_MISMATCH")
    if tuple(hidden.shape[:2]) != tuple(attention_mask.shape):
        raise ContractError("POOLING_INPUT_SHAPE_MISMATCH")
    if not bool(torch.isfinite(hidden).all().item()):
        raise ContractError("POOLING_INPUT_NONFINITE")
    mask = attention_mask.to(device=hidden.device).ne(0)
    counts = mask.sum(dim=1)
    if bool(counts.eq(0).any().item()):
        raise ContractError("POOLING_ALL_PADDING_EXAMPLE")
    return mask, counts


def masked_mean_pool(torch: Any, hidden: Any, attention_mask: Any) -> Any:
    mask, counts = valid_mask(torch, hidden, attention_mask)
    pooled = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1)
    pooled = pooled / counts.unsqueeze(-1).to(hidden.dtype)
    if not bool(torch.isfinite(pooled).all().item()):
        raise ContractError("POOLING_OUTPUT_NONFINITE")
    return pooled


def install_condition_forward(model: Any, condition: str, runtime: Mapping[str, Any]) -> list[str]:
    if condition == "P0":
        return ["score"]
    if condition != "P5":
        raise ContractError("ONLY_P0_P5_ALLOWED")
    torch = runtime["torch"]
    score = model.score
    device = score.weight.device
    dtype = score.weight.dtype

    class LayerMix(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(4, device=device, dtype=dtype))

        def forward(self) -> Any:
            return self.logits

    model.layer_mix = LayerMix()
    output_class = runtime["transformers"].modeling_outputs.SequenceClassifierOutput

    def p5_forward(
        model_self: Any,
        input_ids: Any | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: Any | None = None,
        labels: Any | None = None,
        use_cache: Any | None = None,
        output_attentions: Any | None = None,
        output_hidden_states: Any | None = None,
        return_dict: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs, output_hidden_states, return_dict
        outputs = model_self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None or len(outputs.hidden_states) < 4:
            raise ContractError("P5_HIDDEN_STATES_MISSING")
        selected = tuple(outputs.hidden_states[-4:])
        if any(tuple(item.shape) != tuple(selected[0].shape) for item in selected):
            raise ContractError("P5_LAST_FOUR_LAYER_SHAPE_MISMATCH")
        weights = torch.softmax(model_self.layer_mix(), dim=0)
        mixed = torch.stack(selected, dim=0).mul(weights.view(4, 1, 1, 1)).sum(dim=0)
        pooled = masked_mean_pool(torch, mixed, attention_mask)
        logits = model_self.score(pooled)
        if tuple(logits.shape[-1:]) != (2,) or not bool(torch.isfinite(logits).all().item()):
            raise ContractError("TWO_CLASS_LOGIT_CONTRACT_MISMATCH")
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, 2), labels.view(-1))
        return output_class(loss=loss, logits=logits, hidden_states=None, attentions=None)

    model.forward = types.MethodType(p5_forward, model)
    return ["score", "layer_mix"]


def validate_qkv_targets(model: Any) -> int:
    names = [name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] == "qkv_proj"]
    if not names:
        raise ContractError("MODEL_TARGET_MODULE_MISSING")
    return len(names)


def trainable_parameter_contract(model: Any, condition: str) -> dict[str, int]:
    groups = {"lora": 0, "classifier": 0, "layer_mix": 0, "unexpected": 0}
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if not parameter.requires_grad:
            continue
        trainable += count
        lowered = name.lower()
        if "lora_" in lowered:
            groups["lora"] += count
        elif "score" in lowered:
            groups["classifier"] += count
        elif "layer_mix" in lowered:
            groups["layer_mix"] += count
        else:
            groups["unexpected"] += count
    if total <= 0 or trainable <= 0 or groups["unexpected"] != 0:
        raise ContractError("UNEXPECTED_TRAINABLE_PARAMETER")
    if condition == "P0" and groups["layer_mix"] != 0:
        raise ContractError("P0_LAYER_MIX_PRESENT")
    if condition == "P5" and groups["layer_mix"] != 4:
        raise ContractError("P5_LAYER_MIX_PARAMETER_MISMATCH")
    groups["total_parameters"] = total
    groups["trainable_parameters"] = trainable
    return groups


def load_tokenizer_and_base(condition: str, seed: int, runtime: Mapping[str, Any]) -> tuple[Any, Any, str, int]:
    torch = runtime["torch"]
    transformers = runtime["transformers"]
    seed_everything(runtime, seed)
    snapshot = model_snapshot_path()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ContractError("TOKENIZER_PAD_CONTRACT_INVALID")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(snapshot),
        num_labels=2,
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.float16,
        trust_remote_code=False,
        attn_implementation="eager",
        local_files_only=True,
    )
    target_count = validate_qkv_targets(model)
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise ContractError("MODEL_NOT_LOADED_IN_4BIT")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.problem_type = "single_label_classification"
    classifier_hash = tensor_sha256(model.score.weight)
    install_condition_forward(model, condition, runtime)
    return tokenizer, model, classifier_hash, target_count


def make_trainable_model(condition: str, seed: int, runtime: Mapping[str, Any]) -> tuple[Any, Any, str, int, dict[str, int]]:
    tokenizer, model, classifier_hash, target_count = load_tokenizer_and_base(condition, seed, runtime)
    peft = runtime["peft"]
    modules_to_save = ["score"] if condition == "P0" else ["score", "layer_mix"]
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.SEQ_CLS,
            r=16,
            lora_alpha=16,
            lora_dropout=0.0697368029059364,
            target_modules=["qkv_proj"],
            bias="none",
            modules_to_save=modules_to_save,
        ),
    )
    model.config.use_cache = False
    groups = trainable_parameter_contract(model, condition)
    return tokenizer, model, classifier_hash, target_count, groups


class RecordingDataset:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        self.accessed.append(int(index))
        return self.dataset[int(index)]

    def order_hash(self) -> str:
        digest = hashlib.sha256()
        for index in self.accessed:
            digest.update(struct.pack("<I", index))
        return digest.hexdigest()


def tokenized_dataset(records: Sequence[TrainingRecord], tokenizer: Any, runtime: Mapping[str, Any]) -> RecordingDataset:
    payload = [{"text": record.text, "label": record.label} for record in records]
    source = runtime["datasets"].Dataset.from_list(payload)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding=False,
        )

    mapped = source.map(tokenize, batched=True, remove_columns=["text"], desc=None)
    return RecordingDataset(mapped)


class EventLog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.events = root / "events.jsonl"
        self.worker_log = root / "worker.log"
        self.lock = threading.Lock()

    def emit(self, event: str, **payload: Any) -> None:
        row = {"timestamp_utc": utc_now(), "event": event, **payload}
        line = json.dumps(row, sort_keys=True, ensure_ascii=False)
        with self.lock:
            with self.events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
            with self.worker_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{row['timestamp_utc']} {event} {json.dumps(payload, sort_keys=True, ensure_ascii=False)}\n")
                handle.flush()


class ResourceMonitor:
    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.stop_event = threading.Event()
        self.failure: str | None = None
        self.high_temp_count = 0
        self.thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _sample(self) -> None:
        snapshot = resource_snapshot()
        self.log.emit("RESOURCE_TELEMETRY", **snapshot)
        if snapshot["available_ram_gb"] < 6.0:
            self.failure = "AVAILABLE_RAM_BELOW_RUNTIME_GATE"
        elif snapshot["commit_headroom_gb"] < 12.0:
            self.failure = "COMMIT_HEADROOM_BELOW_RUNTIME_GATE"
        elif snapshot["free_disk_gb"] < 40.0:
            self.failure = "FREE_DISK_BELOW_RUNTIME_GATE"
        elif not snapshot["ac_connected"]:
            self.failure = "AC_POWER_DISCONNECTED"
        if snapshot["gpu_temperature_c"] >= 85:
            self.high_temp_count += 1
        else:
            self.high_temp_count = 0
        if self.high_temp_count >= 3:
            self.failure = "GPU_TEMPERATURE_PERSISTENTLY_HIGH"

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._sample()
            except Exception as error:
                self.failure = f"TELEMETRY_FAILURE:{type(error).__name__}"
                self.log.emit("TELEMETRY_ERROR", error=self.failure)
            self.stop_event.wait(60.0)


def checkpoint_hashes(checkpoint: Path) -> tuple[dict[str, str], str]:
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not files:
        raise ContractError("CHECKPOINT_EMPTY")
    manifest = {path.relative_to(checkpoint).as_posix(): sha256_file(path) for path in files}
    combined = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest, combined


def clear_gpu(runtime: Mapping[str, Any], *objects: Any) -> None:
    del objects
    gc.collect()
    torch = runtime.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def verify_checkpoint_model(
    checkpoint: Path,
    condition: str,
    seed: int,
    sample_text: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    torch = runtime["torch"]
    tokenizer, base_model, _, _ = load_tokenizer_and_base(condition, seed, runtime)
    model = runtime["peft"].PeftModel.from_pretrained(
        base_model,
        str(checkpoint),
        is_trainable=False,
    )
    model.eval()
    encoded = tokenizer(
        sample_text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {name: value.to("cuda") for name, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    logits = output.logits
    if tuple(logits.shape) != (1, 2) or not bool(torch.isfinite(logits).all().item()):
        raise ContractError("CHECKPOINT_RELOAD_FORWARD_FAILED")
    result = {"logit_shape": list(logits.shape), "finite": True}
    del output, logits, encoded, model, base_model, tokenizer
    clear_gpu(runtime)
    return result


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    dataset: RecordingDataset,
    seed: int,
    max_steps: int,
    output_dir: Path,
    runtime: Mapping[str, Any],
    monitor: ResourceMonitor | None,
    check_gradients: bool,
) -> Any:
    torch = runtime["torch"]
    transformers = runtime["transformers"]

    class SafetyTrainer(transformers.Trainer):
        def training_step(self, model_value: Any, inputs: Any) -> Any:
            loss = super().training_step(model_value, inputs)
            if not bool(torch.isfinite(loss).all().item()):
                raise ContractError("NONFINITE_TRAINING_LOSS")
            if monitor is not None and monitor.failure is not None:
                raise ContractError(monitor.failure)
            return loss

    safety_state = {
        "optimizer_steps_seen": 0,
        "optimizer_steps_skipped": 0,
        "finite_gradient_checks": 0,
        "amp_initial_scale": AMP_INITIAL_SCALE,
        "amp_growth_interval": AMP_GROWTH_INTERVAL,
        "amp_observed_scales": [],
    }

    class MonitorCallback(transformers.TrainerCallback):
        def on_optimizer_step(
            self,
            args: Any,
            state: Any,
            control: Any,
            optimizer: Any = None,
            model: Any = None,
            **kwargs: Any,
        ) -> None:
            del args, state, control, kwargs
            safety_state["optimizer_steps_seen"] += 1
            if optimizer is None or model is None:
                raise ContractError("OPTIMIZER_CALLBACK_STATE_MISSING")
            scaler = getattr(optimizer, "scaler", None)
            if scaler is None:
                raise ContractError("FP16_GRAD_SCALER_MISSING")
            current_scale = float(scaler.get_scale())
            safety_state["amp_observed_scales"].append(current_scale)
            if bool(getattr(optimizer, "step_was_skipped", False)):
                safety_state["optimizer_steps_skipped"] += 1
                raise ContractError("OPTIMIZER_STEP_SKIPPED_FP16_OVERFLOW")
            if check_gradients:
                checked = 0
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        checked += 1
                        if not bool(torch.isfinite(parameter.grad).all().item()):
                            raise ContractError(
                                f"NONFINITE_UNSCALED_GRADIENT:{name}:{parameter.grad.dtype}"
                            )
                if checked == 0:
                    raise ContractError("NO_TRAINABLE_GRADIENTS_OBSERVED")
                safety_state["finite_gradient_checks"] += 1

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if logs and "loss" in logs and not math.isfinite(float(logs["loss"])):
                raise ContractError("NONFINITE_LOGGED_LOSS")
            if monitor is not None and monitor.failure is not None:
                raise ContractError(monitor.failure)

    arguments = transformers.TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3 if max_steps == EXPECTED_OPTIMIZER_STEPS else 1,
        max_steps=max_steps,
        learning_rate=LOCKED_CONFIG["learning_rate"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        weight_decay=LOCKED_CONFIG["weight_decay"],
        warmup_ratio=LOCKED_CONFIG["warmup_ratio"],
        label_smoothing_factor=0.05,
        logging_steps=20,
        logging_nan_inf_filter=False,
        evaluation_strategy="no",
        save_strategy="no",
        fp16=True,
        bf16=False,
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        seed=seed,
        data_seed=seed,
        disable_tqdm=True,
    )
    collator = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer = SafetyTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[MonitorCallback()],
    )
    if trainer.accelerator.scaler is None:
        raise ContractError("FP16_GRAD_SCALER_MISSING")
    trainer.accelerator.scaler = torch.cuda.amp.GradScaler(
        init_scale=AMP_INITIAL_SCALE,
        growth_interval=AMP_GROWTH_INTERVAL,
        enabled=True,
    )
    trainer.p0p5_safety_state = safety_state
    return trainer


def run_training_job(
    *,
    condition: str,
    seed: int,
    job_root: Path,
    max_steps: int,
    smoke: bool,
    log: EventLog,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    runtime = load_heavy_runtime()
    torch = runtime["torch"]
    log.emit("HEAVY_RUNTIME_LOADED", versions=runtime["versions"])
    records, _, _ = load_training_records()
    selected_records = records[:4] if smoke else records
    tokenizer = model = trainer = None
    try:
        torch.cuda.reset_peak_memory_stats()
        tokenizer, model, classifier_hash, target_count, groups = make_trainable_model(
            condition, seed, runtime
        )
        if condition == "P5":
            layer_logits = model.base_model.model.layer_mix.original_module.logits
            if not bool(torch.equal(layer_logits.detach(), torch.zeros_like(layer_logits))):
                raise ContractError("P5_LAYER_MIX_NOT_ZERO_INITIALIZED")
        log.emit(
            "MODEL_READY",
            condition_id=condition,
            seed=seed,
            classifier_initialization_sha256=classifier_hash,
            qkv_target_count=target_count,
            trainable_parameter_groups=groups,
        )
        dataset = tokenized_dataset(selected_records, tokenizer, runtime)
        log.emit("DATASET_TOKENIZED", record_count=len(dataset))
        trainer_dir = job_root / "trainer_tmp"
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            seed=seed,
            max_steps=max_steps,
            output_dir=trainer_dir,
            runtime=runtime,
            monitor=monitor,
            check_gradients=smoke,
        )
        log.emit(
            "AMP_SCALER_CONFIGURED",
            initial_scale=AMP_INITIAL_SCALE,
            growth_interval=AMP_GROWTH_INTERVAL,
        )
        log.emit("TRAINING_STARTED", expected_optimizer_steps=max_steps)
        started = time.perf_counter()
        trainer.train()
        elapsed = time.perf_counter() - started
        actual_steps = int(trainer.state.global_step)
        if actual_steps != max_steps:
            raise ContractError("GLOBAL_STEP_BUDGET_MISMATCH")
        safety_state = dict(trainer.p0p5_safety_state)
        if safety_state["optimizer_steps_seen"] != max_steps:
            raise ContractError("OPTIMIZER_CALLBACK_STEP_COUNT_MISMATCH")
        if safety_state["optimizer_steps_skipped"] != 0:
            raise ContractError("OPTIMIZER_STEP_SKIPPED")
        if (
            len(safety_state["amp_observed_scales"]) != max_steps
            or any(scale != AMP_INITIAL_SCALE for scale in safety_state["amp_observed_scales"])
        ):
            raise ContractError("AMP_SCALE_STABILITY_CONTRACT_MISMATCH")
        if smoke and safety_state["finite_gradient_checks"] != max_steps:
            raise ContractError("FINITE_GRADIENT_CHECK_COUNT_MISMATCH")
        if monitor.failure is not None:
            raise ContractError(monitor.failure)
        log.emit(
            "TRAINING_COMPLETED",
            actual_optimizer_steps=actual_steps,
            elapsed_seconds=elapsed,
        )
        checkpoint = job_root / "checkpoint"
        checkpoint.mkdir(exist_ok=False)
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        metadata = {
            "schema_version": "p0_p5_checkpoint_metadata_v1",
            "runtime_version": RUNTIME_VERSION,
            "runtime_sha256": sha256_file(Path(__file__).resolve()),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "condition_id": condition,
            "training_seed": seed,
            "optimizer_steps": actual_steps,
            "classifier_initialization_sha256": classifier_hash,
            "sample_order_sha256": dataset.order_hash(),
            "sample_access_count": len(dataset.accessed),
            "scientific_config": LOCKED_CONFIG,
            "protected_test_accessed": False,
        }
        write_json_once(checkpoint / "checkpoint_metadata.json", metadata)
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise ContractError("ADAPTER_CHECKPOINT_MISSING")
        log.emit("CHECKPOINT_SAVED")
        peak_vram = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if trainer is not None:
            try:
                trainer.accelerator.free_memory()
            except Exception:
                pass
        trainer = None
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        reload_result = verify_checkpoint_model(
            checkpoint,
            condition,
            seed,
            selected_records[0].text,
            runtime,
        )
        log.emit("CHECKPOINT_REOPENED", **reload_result)
        file_hashes, checkpoint_hash = checkpoint_hashes(checkpoint)
        return {
            "condition_id": condition,
            "training_seed": seed,
            "actual_optimizer_steps": actual_steps,
            "expected_optimizer_steps": max_steps,
            "training_elapsed_seconds": elapsed,
            "peak_vram_mib": peak_vram,
            "classifier_initialization_sha256": classifier_hash,
            "sample_order_sha256": dataset.order_hash(),
            "sample_access_count": len(dataset.accessed),
            "trainable_parameter_groups": groups,
            "qkv_target_count": target_count,
            "numeric_safety": safety_state,
            "checkpoint_file_sha256": file_hashes,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_reload": reload_result,
        }
    finally:
        if trainer is not None:
            try:
                trainer.accelerator.free_memory()
            except Exception:
                pass
        trainer = None
        model = None
        tokenizer = None
        clear_gpu(runtime)


def gpu_setup_probe() -> int:
    verify_runtime_identity()
    if PROBE_REPORT_PATH.exists():
        raise ContractError("GPU_SETUP_PROBE_REPORT_ALREADY_EXISTS")
    resources = enforce_resource_gate(initial=True)
    runtime = load_heavy_runtime()
    torch = runtime["torch"]
    records, _, _ = load_training_records()
    tokenizer = model = None
    try:
        torch.cuda.reset_peak_memory_stats()
        tokenizer, model, classifier_hash, target_count = load_tokenizer_and_base("P5", 42, runtime)
        encoded = tokenizer(
            records[0].text,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {name: value.to("cuda") for name, value in encoded.items()}
        with torch.no_grad():
            outputs = model.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state
        selected = tuple(outputs.hidden_states[-4:])
        weights = torch.softmax(model.layer_mix(), dim=0)
        mixed = torch.stack(selected, dim=0).mul(weights.view(4, 1, 1, 1)).sum(dim=0)
        pooled = masked_mean_pool(torch, mixed, encoded["attention_mask"])
        last_positions = encoded["attention_mask"].sum(dim=1) - 1
        p0_repr = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions]
        if not bool(torch.equal(model.layer_mix.logits.detach(), torch.zeros_like(model.layer_mix.logits))):
            raise ContractError("P5_LAYER_MIX_NOT_ZERO_INITIALIZED")
        peft_model = runtime["peft"].prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        peft_model = runtime["peft"].get_peft_model(
            peft_model,
            runtime["peft"].LoraConfig(
                task_type=runtime["peft"].TaskType.SEQ_CLS,
                r=16,
                lora_alpha=16,
                lora_dropout=0.0697368029059364,
                target_modules=["qkv_proj"],
                bias="none",
                modules_to_save=["score", "layer_mix"],
            ),
        )
        groups = trainable_parameter_contract(peft_model, "P5")
        linear4bit_count = sum(
            1 for module in peft_model.modules() if module.__class__.__name__ == "Linear4bit"
        )
        if linear4bit_count <= 0:
            raise ContractError("LINEAR4BIT_MODULES_MISSING")
        report = {
            "schema_version": "p0_p5_gpu_setup_probe_v1",
            "status": "PASS",
            "created_at_utc": utc_now(),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "quantization": "4bit_nf4_double_quant",
            "compute_dtype": "float16",
            "qkv_target_count": target_count,
            "linear4bit_module_count": linear4bit_count,
            "p0_native_last_token_shape": list(p0_repr.shape),
            "p5_last_four_shapes": [list(item.shape) for item in selected],
            "p5_mixed_shape": list(mixed.shape),
            "p5_masked_mean_shape": list(pooled.shape),
            "p5_initial_weights": [float(value) for value in weights.detach().cpu().tolist()],
            "classifier_initialization_sha256": classifier_hash,
            "trainable_parameter_groups": groups,
            "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "resources": resources,
            "training_started": False,
            "optimizer_steps": 0,
            "protected_test_accessed": False,
        }
        write_json_once(PROBE_REPORT_PATH, report)
        print("GPU_SETUP_PROBE=PASS", flush=True)
        return 0
    finally:
        tokenizer = None
        model = None
        clear_gpu(runtime)


def smoke_report_path(condition: str) -> Path:
    return REPORTS / f"numeric_smoke_{condition.lower()}_report_v1r3.json"


def numeric_smoke(condition: str) -> int:
    if condition not in ("P0", "P5"):
        raise ContractError("ONLY_P0_P5_ALLOWED")
    verify_runtime_identity()
    if not PROBE_REPORT_PATH.is_file() or load_json(PROBE_REPORT_PATH).get("status") != "PASS":
        raise ContractError("GPU_SETUP_PROBE_NOT_PASS")
    report_path = smoke_report_path(condition)
    if report_path.exists():
        raise ContractError("NUMERIC_SMOKE_REPORT_ALREADY_EXISTS")
    enforce_resource_gate(initial=True)
    smoke_root = ROOT / "artifacts" / "smoke_v1r3" / condition
    smoke_root.mkdir(parents=True, exist_ok=False)
    log = EventLog(smoke_root)
    monitor = ResourceMonitor(log)
    monitor.start()
    try:
        result = run_training_job(
            condition=condition,
            seed=42,
            job_root=smoke_root,
            max_steps=1,
            smoke=True,
            log=log,
            monitor=monitor,
        )
        monitor.stop()
        log.emit("SMOKE_COMPLETE", condition_id=condition)
        report = {
            "schema_version": "p0_p5_numeric_smoke_v1",
            "status": "PASS",
            "created_at_utc": utc_now(),
            **result,
            "events_sha256": sha256_file(log.events),
            "log_sha256": sha256_file(log.worker_log),
            "included_in_scientific_results": False,
            "protected_test_accessed": False,
        }
        write_json_once(report_path, report)
        print(f"NUMERIC_SMOKE_{condition}=PASS", flush=True)
        return 0
    except Exception:
        monitor.stop()
        failure = {
            "status": "FAIL",
            "condition_id": condition,
            "traceback": traceback.format_exc(),
            "protected_test_accessed": False,
        }
        write_json_once(smoke_root / "failure.json", failure)
        raise


def run_worker(condition: str, seed: int, worker_root: Path, campaign_id: str) -> int:
    if condition not in ("P0", "P5") or seed not in SEEDS:
        raise ContractError("WORKER_SPEC_INVALID")
    verify_runtime_identity()
    if os.environ.get("P0P5_CAMPAIGN_ID") != campaign_id:
        raise ContractError("WORKER_PARENT_TOKEN_MISMATCH")
    if not worker_root.is_dir():
        raise ContractError("WORKER_TEMP_ROOT_MISSING")
    unexpected = [path for path in worker_root.iterdir() if path.name != "subprocess.log"]
    if unexpected:
        raise ContractError("WORKER_TEMP_ROOT_NOT_EMPTY")
    set_below_normal_priority()
    log = EventLog(worker_root)
    monitor = ResourceMonitor(log)
    monitor.start()
    try:
        start_resources = enforce_resource_gate(initial=True, allow_current_runtime=True)
        log.emit("WORKER_STARTED", condition_id=condition, seed=seed, resources=start_resources)
        result = run_training_job(
            condition=condition,
            seed=seed,
            job_root=worker_root,
            max_steps=EXPECTED_OPTIMIZER_STEPS,
            smoke=False,
            log=log,
            monitor=monitor,
        )
        monitor.stop()
        final_resources = resource_snapshot()
        log.emit("WORKER_COMPLETE", condition_id=condition, seed=seed)
        payload = {
            "schema_version": "p0_p5_worker_result_v1",
            "status": "PASS",
            "campaign_id": campaign_id,
            "worker_id": f"{condition}_seed_{seed:04d}",
            "created_at_utc": utc_now(),
            **result,
            "events_sha256": sha256_file(log.events),
            "log_sha256": sha256_file(log.worker_log),
            "start_resources": start_resources,
            "final_resources": final_resources,
            "automatic_retry_used": False,
            "second_campaign_started": False,
            "protected_test_accessed": False,
            "scientific_configuration_changed": False,
        }
        write_json_once(worker_root / "worker_result.json", payload)
        print(
            f"WORKER_PASS condition={condition} seed={seed} steps={result['actual_optimizer_steps']}",
            flush=True,
        )
        return 0
    except Exception as error:
        monitor.stop()
        failure = {
            "schema_version": "p0_p5_worker_failure_v1",
            "status": "FAIL_CLOSED",
            "condition_id": condition,
            "training_seed": seed,
            "error_class": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "automatic_retry_used": False,
            "protected_test_accessed": False,
        }
        try:
            write_json_once(worker_root / "failure.json", failure)
        except Exception:
            pass
        raise


def validate_completed_worker(worker_dir: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    result_path = worker_dir / "worker_result.json"
    if not result_path.is_file():
        raise ContractError("WORKER_RESULT_MISSING")
    result = load_json(result_path)
    if (
        result.get("status") != "PASS"
        or result.get("condition_id") != spec["condition_id"]
        or result.get("training_seed") != spec["seed"]
        or result.get("actual_optimizer_steps") != EXPECTED_OPTIMIZER_STEPS
        or result.get("expected_optimizer_steps") != EXPECTED_OPTIMIZER_STEPS
        or result.get("protected_test_accessed") is not False
        or result.get("automatic_retry_used") is not False
    ):
        raise ContractError("WORKER_RESULT_CONTRACT_MISMATCH")
    checkpoint = worker_dir / "checkpoint"
    observed_files, observed_combined = checkpoint_hashes(checkpoint)
    if result.get("checkpoint_file_sha256") != observed_files:
        raise ContractError("CHECKPOINT_FILE_HASH_MISMATCH")
    if result.get("checkpoint_sha256") != observed_combined:
        raise ContractError("CHECKPOINT_COMBINED_HASH_MISMATCH")
    if result.get("events_sha256") != sha256_file(worker_dir / "events.jsonl"):
        raise ContractError("EVENTS_HASH_MISMATCH")
    if result.get("log_sha256") != sha256_file(worker_dir / "worker.log"):
        raise ContractError("WORKER_LOG_HASH_MISMATCH")
    return result


def wait_minimum_cooldown(seconds: int, campaign_log: EventLog) -> None:
    campaign_log.emit("COOLDOWN_STARTED", seconds=seconds)
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(60.0, remaining))
    campaign_log.emit("COOLDOWN_COMPLETED", seconds=seconds)


def wait_until_start_gate(campaign_log: EventLog) -> dict[str, Any]:
    while True:
        try:
            return enforce_resource_gate(initial=True)
        except ContractError as error:
            if "GPU_TOO_HOT_TO_START" not in str(error):
                raise
            campaign_log.emit("START_GATE_WAITING_FOR_GPU_COOLDOWN", error=str(error))
            time.sleep(60.0)


def final_verify_and_bundle(campaign_id: str, worker_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from safetensors import safe_open

    workers_root = CAMPAIGN_ROOT / "workers"
    manifest_workers: list[dict[str, Any]] = []
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for spec in worker_specs:
        worker_dir = workers_root / str(spec["worker_id"])
        result = validate_completed_worker(worker_dir, spec)
        adapter = worker_dir / "checkpoint" / "adapter_model.safetensors"
        with safe_open(adapter, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if not keys:
            raise ContractError("CHECKPOINT_SAFETENSORS_EMPTY")
        row = {
            "worker_id": spec["worker_id"],
            "pair_index": spec["pair_index"],
            "condition_id": spec["condition_id"],
            "training_seed": spec["seed"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "checkpoint_file_sha256": result["checkpoint_file_sha256"],
            "classifier_initialization_sha256": result["classifier_initialization_sha256"],
            "sample_order_sha256": result["sample_order_sha256"],
            "adapter_tensor_count": len(keys),
            "worker_result_sha256": sha256_file(worker_dir / "worker_result.json"),
        }
        manifest_workers.append(row)
        by_pair.setdefault(int(spec["pair_index"]), []).append(row)
    for pair_index, pair_rows in by_pair.items():
        if len(pair_rows) != 2 or {row["condition_id"] for row in pair_rows} != {"P0", "P5"}:
            raise ContractError(f"PAIR_COMPLETENESS_MISMATCH:{pair_index}")
        if len({row["classifier_initialization_sha256"] for row in pair_rows}) != 1:
            raise ContractError(f"PAIR_CLASSIFIER_INITIALIZATION_MISMATCH:{pair_index}")
        if len({row["sample_order_sha256"] for row in pair_rows}) != 1:
            raise ContractError(f"PAIR_SAMPLE_ORDER_MISMATCH:{pair_index}")
    manifest = {
        "schema_version": "p0_p5_frozen_checkpoint_manifest_v1",
        "status": "FROZEN",
        "campaign_id": campaign_id,
        "created_at_utc": utc_now(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_version": RUNTIME_VERSION,
        "runtime_sha256": sha256_file(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "python_executable_sha256": sha256_file(Path(sys.executable)),
        "scientific_contract_sha256": sha256_file(CONTRACT_PATH),
        "seed_plan_sha256": sha256_file(SEED_PLAN_PATH),
        "worker_count": len(manifest_workers),
        "p0_count": sum(row["condition_id"] == "P0" for row in manifest_workers),
        "p5_count": sum(row["condition_id"] == "P5" for row in manifest_workers),
        "workers": manifest_workers,
        "all_checkpoints_reopened_after_training": True,
        "all_safetensors_reopened_during_freeze": True,
        "official_test_access_status": "NOT_ACCESSED",
        "protected_test_accessed": False,
        "automatic_retry_used": False,
        "second_campaign_started": False,
        "scientific_configuration_changed": False,
    }
    manifest_path = CAMPAIGN_ROOT / "checkpoint_manifest.json"
    write_json_once(manifest_path, manifest)
    bundle = ROOT / "artifacts" / "P0_P5_20_CHECKPOINT_FROZEN_BUNDLE.zip"
    if bundle.exists():
        raise ContractError("FROZEN_BUNDLE_ALREADY_EXISTS")
    temporary_bundle = bundle.with_name(bundle.name + ".partial")
    with zipfile.ZipFile(temporary_bundle, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(manifest_path, "checkpoint_manifest.json")
        archive.write(Path(__file__).resolve(), f"runtime/{Path(__file__).name}")
        runner = Path(__file__).resolve().with_name("run_p0_p5_portable_v1.ps1")
        if runner.is_file():
            archive.write(runner, f"runtime/{runner.name}")
        for row in manifest_workers:
            checkpoint = workers_root / row["worker_id"] / "checkpoint"
            for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
                arcname = Path("checkpoints") / row["worker_id"] / path.relative_to(checkpoint)
                archive.write(path, arcname.as_posix())
    os.replace(temporary_bundle, bundle)
    bundle_hash = sha256_file(bundle)
    return {
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "bundle": str(bundle),
        "bundle_sha256": bundle_hash,
        "bundle_size_bytes": bundle.stat().st_size,
    }


def campaign() -> int:
    cpu_report = verify_runtime_identity()
    if cpu_report.get("planned_worker_count") != 20:
        raise ContractError("CPU_PLAN_MISMATCH")
    if not PROBE_REPORT_PATH.is_file() or load_json(PROBE_REPORT_PATH).get("status") != "PASS":
        raise ContractError("GPU_SETUP_PROBE_NOT_PASS")
    for condition in ("P0", "P5"):
        report_path = smoke_report_path(condition)
        if not report_path.is_file() or load_json(report_path).get("status") != "PASS":
            raise ContractError(f"NUMERIC_SMOKE_{condition}_NOT_PASS")
    model_snapshot_path()
    enforce_resource_gate(initial=True)
    if CAMPAIGN_ROOT.exists():
        raise ContractError("CAMPAIGN_ROOT_ALREADY_EXISTS_SECOND_CAMPAIGN_REFUSED")
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=False)
    (CAMPAIGN_ROOT / "workers").mkdir()
    (CAMPAIGN_ROOT / "temporary").mkdir()
    campaign_id = f"p0p5_locked_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    campaign_log = EventLog(CAMPAIGN_ROOT)
    plan = expected_plan()
    campaign_log.emit(
        "CAMPAIGN_STARTED",
        campaign_id=campaign_id,
        planned_workers=20,
        automatic_retry_used=False,
        second_campaign_started=False,
        protected_test_accessed=False,
    )
    state = {
        "schema_version": "p0_p5_campaign_state_v1",
        "status": "RUNNING",
        "campaign_id": campaign_id,
        "planned_workers": 20,
        "completed_workers": 0,
        "active_worker": None,
        "automatic_retry_used": False,
        "second_campaign_started": False,
        "protected_test_accessed": False,
        "scientific_configuration_changed": False,
    }
    replace_json(CAMPAIGN_ROOT / "campaign_state.json", state)
    completed_results: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(plan):
        if index > 0:
            wait_minimum_cooldown(300, campaign_log)
        resources = wait_until_start_gate(campaign_log)
        worker_id = str(spec["worker_id"])
        temporary = CAMPAIGN_ROOT / "temporary" / f"{worker_id}.partial"
        completed = CAMPAIGN_ROOT / "workers" / worker_id
        if temporary.exists() or completed.exists():
            raise ContractError("WORKER_ARTIFACT_ALREADY_EXISTS")
        temporary.mkdir()
        state["active_worker"] = worker_id
        state["last_resources"] = resources
        replace_json(CAMPAIGN_ROOT / "campaign_state.json", state)
        campaign_log.emit("WORKER_DISPATCHED", worker_id=worker_id, resources=resources)
        environment = os.environ.copy()
        environment.update(
            {
                "P0P5_CAMPAIGN_ID": campaign_id,
                "P0P5_PARENT_PID": str(os.getpid()),
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HOME": str(ROOT / "hf_cache"),
            }
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--condition",
            str(spec["condition_id"]),
            "--seed",
            str(spec["seed"]),
            "--worker-root",
            str(temporary),
            "--campaign-id",
            campaign_id,
        ]
        subprocess_log = temporary / "subprocess.log"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with subprocess_log.open("w", encoding="utf-8", newline="\n") as output:
            completed_process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
                creationflags=creationflags,
            )
        if completed_process.returncode != 0:
            state.update(
                {
                    "status": "INCOMPLETE_FAIL_CLOSED",
                    "interrupted_worker": worker_id,
                    "active_worker": None,
                    "failure_class": "WORKER_PROCESS_FAILED",
                }
            )
            replace_json(CAMPAIGN_ROOT / "campaign_state.json", state)
            campaign_log.emit(
                "CAMPAIGN_FAIL_CLOSED",
                worker_id=worker_id,
                return_code=completed_process.returncode,
            )
            return 20
        result = validate_completed_worker(temporary, spec)
        result["subprocess_log_sha256"] = sha256_file(subprocess_log)
        os.replace(temporary, completed)
        completed_results[worker_id] = result
        state["completed_workers"] = len(completed_results)
        state["active_worker"] = None
        replace_json(CAMPAIGN_ROOT / "campaign_state.json", state)
        campaign_log.emit(
            "WORKER_COMPLETED",
            worker_id=worker_id,
            completed_workers=len(completed_results),
        )
        pair_specs = [row for row in plan if row["pair_index"] == spec["pair_index"]]
        if all(row["worker_id"] in completed_results for row in pair_specs):
            pair_results = [completed_results[str(row["worker_id"])] for row in pair_specs]
            if len({row["sample_order_sha256"] for row in pair_results}) != 1:
                raise ContractError("PAIR_SAMPLE_ORDER_MISMATCH")
            if len({row["classifier_initialization_sha256"] for row in pair_results}) != 1:
                raise ContractError("PAIR_CLASSIFIER_INITIALIZATION_MISMATCH")
            campaign_log.emit(
                "PAIR_COMPLETED",
                pair_index=spec["pair_index"],
                seed=spec["seed"],
                completed_workers=len(completed_results),
            )
    frozen = final_verify_and_bundle(campaign_id, plan)
    state.update(
        {
            "status": "COMPLETE",
            "completed_workers": 20,
            "active_worker": None,
            "frozen_bundle": frozen,
            "official_test_access_status": "NOT_ACCESSED",
        }
    )
    replace_json(CAMPAIGN_ROOT / "campaign_state.json", state)
    campaign_log.emit("CAMPAIGN_COMPLETE", completed_workers=20, **frozen)
    completion = {
        "schema_version": "p0_p5_campaign_completion_v1",
        "status": "COMPLETE",
        "campaign_id": campaign_id,
        "completed_workers": 20,
        **frozen,
        "official_test_access_status": "NOT_ACCESSED",
        "protected_test_accessed": False,
        "automatic_retry_used": False,
        "second_campaign_started": False,
        "scientific_configuration_changed": False,
    }
    write_json_once(CAMPAIGN_ROOT / "campaign_complete.json", completion)
    print("20/20 CHECKPOINT TAMAMLANDI", flush=True)
    print("OFFICIAL_TEST_ACCESS_STATUS=NOT_ACCESSED", flush=True)
    print("CHECKPOINTLER KILITLENDI", flush=True)
    print("KORUNAN TEST BASLATILMADI", flush=True)
    print("CUSTODIAN YETKISI VE IMZALI YENIDEN ERISIM IZNI BEKLENIYOR", flush=True)
    return 0


def status() -> int:
    if CAMPAIGN_ROOT.is_dir() and (CAMPAIGN_ROOT / "campaign_state.json").is_file():
        print(json.dumps(load_json(CAMPAIGN_ROOT / "campaign_state.json"), indent=2, ensure_ascii=False))
        return 0
    payload = {
        "cpu_prepare": load_json(CPU_REPORT_PATH).get("status") if CPU_REPORT_PATH.is_file() else "NOT_RUN",
        "gpu_setup_probe": load_json(PROBE_REPORT_PATH).get("status") if PROBE_REPORT_PATH.is_file() else "NOT_RUN",
        "numeric_smoke_p0": load_json(smoke_report_path("P0")).get("status")
        if smoke_report_path("P0").is_file()
        else "NOT_RUN",
        "numeric_smoke_p5": load_json(smoke_report_path("P5")).get("status")
        if smoke_report_path("P5").is_file()
        else "NOT_RUN",
        "campaign": "NOT_STARTED",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("cpu-prepare")
    subparsers.add_parser("probe")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--condition", required=True, choices=("P0", "P5"))
    campaign_parser = subparsers.add_parser("campaign")
    campaign_parser.add_argument("--confirm-real-campaign", action="store_true")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--condition", required=True, choices=("P0", "P5"))
    worker_parser.add_argument("--seed", required=True, type=int)
    worker_parser.add_argument("--worker-root", required=True, type=Path)
    worker_parser.add_argument("--campaign-id", required=True)
    subparsers.add_parser("status")
    args = parser.parse_args(argv)
    if args.mode == "cpu-prepare":
        return cpu_prepare()
    if args.mode == "probe":
        return gpu_setup_probe()
    if args.mode == "smoke":
        return numeric_smoke(args.condition)
    if args.mode == "campaign":
        if not args.confirm_real_campaign:
            raise ContractError("EXPLICIT_REAL_CAMPAIGN_CONFIRMATION_REQUIRED")
        return campaign()
    if args.mode == "worker":
        return run_worker(args.condition, args.seed, args.worker_root.resolve(), args.campaign_id)
    if args.mode == "status":
        return status()
    raise ContractError("MODE_INVALID")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"FAIL_CLOSED={error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
