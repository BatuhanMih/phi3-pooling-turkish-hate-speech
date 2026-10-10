"""Read-only verifier for the immutable P2_EQ4 V1r2 execution package.

The verifier is intentionally CPU-only.  It validates the preserved V1/V1r1
evidence, the V1r2 preparation payloads, the separation of setup/smoke/real
modes, and (when requested) prerequisite reports produced by earlier GPU
stages.  It never loads the model, initializes CUDA, trains, predicts, or
opens test/reserve data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Final, Mapping


RUNTIME_RELATIVE: Final = (
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.py"
)
VERIFIER_RELATIVE: Final = (
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.py"
)
RUNNER_RELATIVE: Final = (
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.ps1"
)
RUNTIME_ROOT_RELATIVE: Final = (
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4"
)
PACKAGE_HELPER_RELATIVE: Final = (
    "tools/check_hpo_phi3_p5_asymmetric_focal_package_origins_v1r2.py"
)
PACKAGE_HELPER_SHA256: Final = (
    "a6445ee30fc9f863bed297d800354adfb2f904923f4f395a189ae9fae5c597c8"
)

V1_V1R1_LOCKS: Final[dict[str, str]] = {
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1.py": (
        "66fd56123dc0294f7296b9b70b35b9e0f9089faeecf517898a452a1ca78bb21e"
    ),
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1.py": (
        "f1e7e8a8e5ea4c0100226494a5e583bcf87278642651a1fcb8fb4893fe987523"
    ),
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1.ps1": (
        "7ba0f750d12ab8cbd9d52df8014d9a23090b0d6bcd23bf4c49070f1e1b5a8ce9"
    ),
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1.py": (
        "cd6f1720de95a42f7e74f5893303aa262bd42544d16db702f7b8ddf4871a8d81"
    ),
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1.py": (
        "c7b86ab3392a6858afb40159805723652075517cd39a00c8e173eafdd898249e"
    ),
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1.ps1": (
        "fda4c65c23afe9bb165a0bbbd50c212240ad5d7af2588bd1d7c9c8f1c8f73caf"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/cpu_equivalence_report.json": (
        "1b7adb31a80da5f69738b7b8959b65bc141e5d758f61c722790a771326cfbe57"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/dry_gate_report.json": (
        "9c031bd9a01fce5a1e11bb8be72fe8a2cb9af32765881523453cd09f278bd4a1"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/scientific_protocol.json": (
        "8899452939cff2626c4f234e52af1729a477caab5cc5cb38ca8295ea74bde85b"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/source_manifest.json": (
        "6c05382301242fe563f8c980dfc1c4184a6c54bbe3b63015d02a704bef9bbace"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/v1_failure_preservation_report.json": (
        "4e9f6e589903c2b72113f91c6123a4b6c72a95f43487a1d6e3b72731407c05de"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/worker_matrix.json": (
        "3e5a8e8b02cca00f0e9727fea2d25604196bcc96201061eacd8b59e5813bb102"
    ),
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2.py": (
        "d274b08c01509fa3c2bd4f1ad5375d956a42d852738aaeec2de098a026eb7532"
    ),
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2.py": (
        "89dfe9ac048e3a2c1b54e0eb70bb93ea9b730760645b125f2b1d25da749ef7f0"
    ),
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2.ps1": (
        "88225a07a1275cbbfe0aa4397f0a6b3deb6b2f1e6997b8f049ea99fef5d02903"
    ),
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r1.py": (
        "9d8a9394dea81898c9dacb959464e2c50fe9a69e26ae0999d89605226d36532e"
    ),
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r1.py": (
        "90f384134e0d1e148129b65979f8312a7e4731911562a544283de29b6a4fb929"
    ),
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r1.ps1": (
        "4f4ad850a3f69c8a2dd9ab81bebf2ae4dc24e255b3b3aa82b3a87b663658f2f6"
    ),
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r2.py": (
        "f2301086e95b7504a9e52d505a935a243992a816a58a59d166d6aefebbda21e6"
    ),
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r2.py": (
        "de4357481b3e5bd5e06bbefe54eb9e9c48b78778ea0d9337987a9e4d0efea0af"
    ),
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r2.ps1": (
        "45b3967203fa9becc84db16203d19de87f1fe0c58adc49985ff79113d42773f3"
    ),
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3.py": (
        "3b5fd03a786a2f32335ad68f019c53d22d0fb86b5b1c71272021c20afe55e6e2"
    ),
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3.py": (
        "f9355e8a8882cf4135908d0de25d0cff2c41664b3e1b6421417f46d0abc35ee4"
    ),
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3.ps1": (
        "9fda37b360f3951aa6de03abba6a000fc761b69f186a22a1577678df88cd06ef"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/cpu_equivalence_report.json": (
        "1b7adb31a80da5f69738b7b8959b65bc141e5d758f61c722790a771326cfbe57"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/dry_gate_report.json": (
        "70849c78ba1361234caa286bb6e53c9d7143ba3d6c905f76e7c095e1a7e47572"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/gpu_execution_protocol.json": (
        "48c1355a59d3af5db352bb45fbb383d88688bb273991d900fa9b30eaca633728"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/scientific_protocol.json": (
        "8899452939cff2626c4f234e52af1729a477caab5cc5cb38ca8295ea74bde85b"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/source_manifest.json": (
        "6c05382301242fe563f8c980dfc1c4184a6c54bbe3b63015d02a704bef9bbace"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/v1r1_preservation_report.json": (
        "76221475f809a415564ac94f6547c4f4c394cc05c2a3a8adfe12230c3025eb6b"
    ),
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/worker_matrix.json": (
        "3e5a8e8b02cca00f0e9727fea2d25604196bcc96201061eacd8b59e5813bb102"
    ),
}

CPU_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "source_manifest.json",
    "scientific_protocol.json",
    "worker_matrix.json",
    "cpu_equivalence_report.json",
    "v1r1_preservation_report.json",
    "gpu_execution_protocol.json",
    "dry_gate_report.json",
)
BYTE_EQUAL_TO_V1R1: Final[tuple[str, ...]] = (
    "source_manifest.json",
    "scientific_protocol.json",
    "worker_matrix.json",
    "cpu_equivalence_report.json",
)
EXPECTED_PARAMETER_CONTRACT: Final[dict[str, Any]] = {
    "total_parameter_count": 1_916_943_360,
    "trainable_parameter_count": 6_297_600,
    "trainable_parameter_groups": {
        "attention_pool": 0,
        "classifier": 6_144,
        "layer_mix": 0,
        "lora": 6_291_456,
    },
}


class VerificationError(RuntimeError):
    """Fail-closed verifier error carrying a stable code."""


def require(condition: bool, code: str, gates: list[dict[str, str]]) -> None:
    if not condition:
        raise VerificationError(code)
    gates.append({"gate": code, "status": "PASS"})


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise VerificationError(f"FILE_MISSING_OR_SYMLINK:{path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise VerificationError(f"JSON_INVALID:{path.name}") from None
    if not isinstance(value, dict):
        raise VerificationError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def import_runtime(path: Path) -> Any:
    heavy = {
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "safetensors",
    }
    if heavy.intersection(sys.modules):
        raise VerificationError("HEAVY_IMPORT_PRESENT_BEFORE_RUNTIME_IMPORT")
    source_dir = str(path.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location(
        "phi3_pooling_fixed4_followup_v1r2_verifier_target", path
    )
    if spec is None or spec.loader is None:
        raise VerificationError("RUNTIME_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if heavy.intersection(sys.modules):
        raise VerificationError("HEAVY_IMPORT_NOT_LAZY")
    return module


def named_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    rows = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(rows) != 1:
        raise VerificationError(f"AST_FUNCTION_COUNT_INVALID:{name}")
    return rows[0]


def call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            parts = [target.attr]
            current = target.value
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            names.add(".".join(reversed(parts)))
    return names


def validate_report_link_hashes(
    report: Mapping[str, Any],
    runtime_hash: str,
    verifier_hash: str,
    dry_hash: str,
    gates: list[dict[str, str]],
    prefix: str,
) -> None:
    aliases = {
        "runtime": ("runtime_sha256", "runtime_source_sha256"),
        "verifier": ("verifier_sha256", "cpu_verifier_sha256"),
        "dry": ("dry_gate_sha256", "dry_gate_report_sha256"),
    }
    expected = {
        "runtime": runtime_hash,
        "verifier": verifier_hash,
        "dry": dry_hash,
    }
    for label, keys in aliases.items():
        values = [report.get(key) for key in keys if key in report]
        require(len(values) == 1, f"{prefix}_{label.upper()}_HASH_LINK_PRESENT", gates)
        require(values[0] == expected[label], f"{prefix}_{label.upper()}_HASH_LINK_EXACT", gates)


def validate_setup_report(
    path: Path,
    runtime_hash: str,
    verifier_hash: str,
    dry_hash: str,
    gates: list[dict[str, str]],
) -> None:
    report = load_json(path)
    exact = {
        "schema_version": "phi3_pooling_fixed4_followup_setup_probe_v1r2r4",
        "status": "PASS",
        "mode": "GPU_SETUP_PROBE",
        "setup_probe_status": "PASS",
        "condition_id": "P2_EQ4",
        "trainer_created": True,
        "model_loaded": True,
        "stop_stage": "trainer_created",
        "forward_started": False,
        "backward_started": False,
        "training_started": False,
        "prediction_started": False,
        "checkpoint_written": False,
        "actual_optimizer_steps": 0,
        "scheduler_steps": 0,
        "official_test_accessed": False,
        "reserve_1_accessed": False,
        "reserve_2_accessed": False,
        "test_dataset_access_count": 0,
    }
    for key, value in exact.items():
        require(report.get(key) == value, f"SETUP_REPORT_EXACT:{key}", gates)
    require(
        report.get("trainable_parameter_groups")
        == EXPECTED_PARAMETER_CONTRACT["trainable_parameter_groups"],
        "SETUP_PARAMETER_GROUPS_EXACT",
        gates,
    )
    validate_report_link_hashes(
        report, runtime_hash, verifier_hash, dry_hash, gates, "SETUP"
    )


def validate_numeric_smoke_report(
    path: Path,
    runtime_hash: str,
    verifier_hash: str,
    dry_hash: str,
    gates: list[dict[str, str]],
) -> None:
    report = load_json(path)
    exact = {
        "schema_version": "phi3_pooling_fixed4_followup_numeric_smoke_v1r2r4",
        "status": "PASS",
        "mode": "GPU_NUMERIC_SMOKE",
        "numeric_smoke_status": "PASS",
        "condition_id": "P2_EQ4",
        "trainer_created": True,
        "model_loaded": True,
        "full_training_started": False,
        "forward_started": True,
        "backward_started": True,
        "prediction_started": False,
        "checkpoint_written": False,
        "actual_optimizer_steps": 0,
        "scheduler_steps": 0,
        "loss_finite": True,
        "logits_finite": True,
        "logits_last_dimension": 2,
        "gradient_finite": True,
        "fixed_layer_weights_trainable": False,
        "official_test_accessed": False,
        "reserve_1_accessed": False,
        "reserve_2_accessed": False,
        "test_dataset_access_count": 0,
    }
    for key, value in exact.items():
        require(report.get(key) == value, f"SMOKE_REPORT_EXACT:{key}", gates)
    groups = report.get("gradient_groups")
    require(isinstance(groups, dict) and set(groups) == {"lora", "classifier"}, "SMOKE_GRADIENT_GROUP_SET", gates)
    for group_name in ("lora", "classifier"):
        group = groups[group_name]
        require(isinstance(group, dict), f"SMOKE_GRADIENT_GROUP_OBJECT:{group_name}", gates)
        tensor_count = group.get("tensor_count")
        absolute_sum = group.get("nonzero_absolute_sum")
        require(isinstance(tensor_count, int) and not isinstance(tensor_count, bool) and tensor_count > 0, f"SMOKE_GRADIENT_TENSORS_POSITIVE:{group_name}", gates)
        require(isinstance(absolute_sum, (int, float)) and not isinstance(absolute_sum, bool) and float(absolute_sum) > 0.0, f"SMOKE_GRADIENT_SUM_POSITIVE:{group_name}", gates)
        require(group.get("finite") is True, f"SMOKE_GRADIENT_FINITE:{group_name}", gates)
    forbidden_payload_keys = {
        "texts",
        "raw_text",
        "examples",
        "input_ids",
        "labels",
        "logits",
        "tensors",
        "predictions",
    }
    require(not forbidden_payload_keys.intersection(report), "SMOKE_NO_RAW_EXAMPLES_OR_TENSORS", gates)
    validate_report_link_hashes(
        report, runtime_hash, verifier_hash, dry_hash, gates, "SMOKE"
    )


def validate_real_report(
    path: Path,
    workers_root: Path,
    runtime_hash: str,
    verifier_hash: str,
    dry_hash: str,
    gates: list[dict[str, str]],
) -> None:
    report = load_json(path)
    exact = {
        "schema_version": "phi3_pooling_fixed4_followup_report_v1r2r4",
        "status": "PASS",
        "evidence_role": "prospectively_locked_follow_up_ablation",
        "condition_id": "P2_EQ4",
        "display_label": "P2",
        "worker_status": "30_OF_30_PASS",
        "new_worker_count": 30,
        "frozen_prior_worker_count": 90,
        "combined_worker_count": 120,
        "analysis_unit": "five_repetitions",
        "binary_significance_gate_applied": False,
        "selection_or_promotion_applied": False,
        "old_and_new_workers_contemporaneous": False,
        "official_test_accessed": False,
        "reserve_1_accessed": False,
        "reserve_2_accessed": False,
        "test_dataset_access_count": 0,
        "automatic_retry_attempted": False,
    }
    for key, value in exact.items():
        require(report.get(key) == value, f"REAL_REPORT_EXACT:{key}", gates)
    require(report.get("parameter_contract") == EXPECTED_PARAMETER_CONTRACT, "REAL_PARAMETER_CONTRACT", gates)
    contrasts = report.get("contrasts")
    require(
        isinstance(contrasts, dict)
        and set(contrasts)
        == {"P5_minus_P2_EQ4", "P2_EQ4_minus_P1", "P1_minus_P0", "P5_minus_P0"},
        "REAL_CONTRAST_SET",
        gates,
    )
    for name, contrast in contrasts.items():
        require(isinstance(contrast, dict), f"REAL_CONTRAST_OBJECT:{name}", gates)
        require(contrast.get("unit") == "repetition" and contrast.get("n") == 5, f"REAL_CONTRAST_N5:{name}", gates)
        require(len(contrast.get("repetition_deltas", [])) == 5, f"REAL_CONTRAST_DELTAS5:{name}", gates)
        require(contrast.get("degrees_of_freedom") == 4, f"REAL_CONTRAST_DF4:{name}", gates)
        require(contrast.get("binary_significance_gate_applied") is False, f"REAL_NO_BINARY_GATE:{name}", gates)
    rows = report.get("worker_results")
    require(isinstance(rows, list) and len(rows) == 30, "REAL_REPORT_30_WORKER_ROWS", gates)
    require([row.get("run_index") for row in rows] == list(range(1, 31)), "REAL_REPORT_RUN_INDICES", gates)
    require(workers_root.is_dir() and not workers_root.is_symlink(), "REAL_WORKERS_ROOT", gates)
    worker_directories = sorted(path for path in workers_root.iterdir() if path.is_dir())
    require(len(worker_directories) == 30, "REAL_EXACT_30_WORKER_DIRECTORIES", gates)
    for row in rows:
        run_index = int(row["run_index"])
        matches = [path for path in worker_directories if path.name.startswith(f"run_{run_index:03d}_P2_EQ4_")]
        require(len(matches) == 1, f"REAL_WORKER_DIRECTORY:{run_index}", gates)
        result_path = matches[0] / "worker_result.json"
        status_path = matches[0] / "worker_status.json"
        result = load_json(result_path)
        status = load_json(status_path)
        require(status.get("status") == "PASS", f"REAL_WORKER_STATUS:{run_index}", gates)
        require(result.get("schema_version") == "phi3_pooling_fixed4_followup_worker_result_v1r2r4", f"REAL_WORKER_SCHEMA:{run_index}", gates)
        require(result.get("run_index") == run_index and result.get("condition_id") == "P2_EQ4", f"REAL_WORKER_ID:{run_index}", gates)
        require(result.get("actual_optimizer_steps") == 564, f"REAL_WORKER_STEPS:{run_index}", gates)
        require(result.get("test_dataset_access_count") == 0 and result.get("final_test_accessed") is False, f"REAL_WORKER_NO_TEST:{run_index}", gates)
        require(sha256_file(result_path) == row.get("worker_result_sha256"), f"REAL_WORKER_HASH:{run_index}", gates)
    validate_report_link_hashes(report, runtime_hash, verifier_hash, dry_hash, gates, "REAL")


def verify(
    project_root: Path,
    runtime_source: Path,
    runner: Path,
    runtime_root: Path,
    required_state: str,
    expected_runtime_sha256: str,
    expected_verifier_sha256: str,
) -> dict[str, Any]:
    verifier_source = project_root / VERIFIER_RELATIVE
    package_helper = project_root / PACKAGE_HELPER_RELATIVE
    gates: list[dict[str, str]] = []

    expected_locations = {
        runtime_source: project_root / RUNTIME_RELATIVE,
        runner: project_root / RUNNER_RELATIVE,
        runtime_root: project_root / RUNTIME_ROOT_RELATIVE,
    }
    for actual, expected in expected_locations.items():
        require(actual.resolve() == expected.resolve(), f"CANONICAL_LOCATION:{expected.name}", gates)
    for path in (runtime_source, verifier_source, runner, package_helper):
        require(path.is_file() and not path.is_symlink(), f"SOURCE_EXISTS:{path.name}", gates)
    require(sha256_file(package_helper) == PACKAGE_HELPER_SHA256, "PACKAGE_HELPER_HASH_LOCKED", gates)
    require(sha256_file(runtime_source) == expected_runtime_sha256, "RUNTIME_HASH_LOCKED", gates)
    require(sha256_file(verifier_source) == expected_verifier_sha256, "VERIFIER_SELF_HASH_LOCKED", gates)

    for relative, expected_hash in V1_V1R1_LOCKS.items():
        require(sha256_file(project_root / relative) == expected_hash, f"PRESERVED_HASH:{relative}", gates)

    v1r1_root = project_root / "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1"
    require(v1r1_root.is_dir() and not v1r1_root.is_symlink(), "V1R1_ROOT_EXISTS", gates)
    require(
        sorted(path.name for path in v1r1_root.iterdir())
        == sorted(
            [
                "source_manifest.json",
                "scientific_protocol.json",
                "worker_matrix.json",
                "cpu_equivalence_report.json",
                "v1_failure_preservation_report.json",
                "dry_gate_report.json",
            ]
        ),
        "V1R1_EXACTLY_SIX_ARTIFACTS",
        gates,
    )

    runtime = import_runtime(runtime_source)
    require(runtime.RUNTIME_VERSION == "hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4", "RUNTIME_VERSION", gates)
    require(Path(runtime.FIXED_PYTHON).resolve() == Path(sys.executable).resolve(), "FIXED_INTERPRETER", gates)
    require(runtime.verify_v1r1_preserved(project_root) == runtime.V1R1_PRESERVATION_SHA256, "V1R1_RUNTIME_PRESERVATION_GATE", gates)
    require(
        runtime.verify_v1r2_failure_preserved(project_root)
        == runtime.V1R2_FAILURE_PRESERVATION_SHA256,
        "V1R2_FAILURE_PRESERVATION_GATE",
        gates,
    )
    require(
        runtime.verify_v1r2r1_failure_preserved(project_root)
        == runtime.V1R2R1_FAILURE_PRESERVATION_SHA256,
        "V1R2R1_FAILURE_PRESERVATION_GATE",
        gates,
    )
    require(
        runtime.verify_v1r2r2_failure_preserved(project_root)
        == runtime.V1R2R2_FAILURE_PRESERVATION_SHA256,
        "V1R2R2_FAILURE_PRESERVATION_GATE",
        gates,
    )
    require(
        runtime.verify_v1r2r3_failure_preserved(project_root)
        == runtime.V1R2R3_FAILURE_PRESERVATION_SHA256,
        "V1R2R3_FAILURE_PRESERVATION_GATE",
        gates,
    )

    require(runtime_root.is_dir() and not runtime_root.is_symlink(), "V1R2_RUNTIME_ROOT_EXISTS", gates)
    artifact_paths = {name: runtime_root / name for name in CPU_ARTIFACT_NAMES}
    for name, path in artifact_paths.items():
        require(path.is_file() and not path.is_symlink(), f"CPU_ARTIFACT_EXISTS:{name}", gates)

    allowed_dynamic = {
        "gpu_setup_probe",
        "gpu_numeric_smoke",
        "workers",
        "p2_eq4_followup_report.json",
        "p2_eq4_followup_incomplete_report.json",
    }
    top_level = {path.name for path in runtime_root.iterdir()}
    require(top_level.issubset(set(CPU_ARTIFACT_NAMES) | allowed_dynamic), "NO_UNRECOGNIZED_RUNTIME_ARTIFACT", gates)

    stored = {name: load_json(path) for name, path in artifact_paths.items()}
    payloads = runtime.build_cpu_payloads(project_root)
    require(isinstance(payloads, dict) and set(payloads) == set(CPU_ARTIFACT_NAMES[:-1]), "CPU_BUILDER_EXACT_SIX_PAYLOADS", gates)
    for name in CPU_ARTIFACT_NAMES[:-1]:
        require(stored[name] == payloads[name], f"CPU_PAYLOAD_REPRODUCES:{name}", gates)
    expected_dry = runtime.build_dry_gate(project_root, payloads)
    require(stored["dry_gate_report.json"] == expected_dry, "DRY_GATE_REPRODUCES", gates)

    for name in BYTE_EQUAL_TO_V1R1:
        prior_path = v1r1_root / name
        require(sha256_file(artifact_paths[name]) == sha256_file(prior_path), f"BYTE_EQUAL_V1R1:{name}", gates)

    preservation = stored["v1r1_preservation_report.json"]
    require(preservation == runtime.build_v1r1_preservation_report(project_root), "PRESERVATION_REPORT_REPRODUCES", gates)
    preservation_exact = {
        "status": "PASS",
        "v1r1_preserved": True,
        "v1r2_failure_preserved": True,
        "v1r2_cpu_create_invocation_count": 1,
        "v1r2_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2_exact_error_code": "PERMISSION_ERROR_WINERROR_5_RUNTIME_ROOT_CREATE",
        "v1r2_runtime_target_created": False,
        "v1r2_artifact_count": 0,
        "v1r2_manual_retry_attempted": False,
        "v1r2r1_failure_preserved": True,
        "v1r2r1_cpu_create_invocation_count": 1,
        "v1r2r1_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2r1_exact_error_code": "ARGPARSE_NAMESPACE_ATTRIBUTE_REVISION_MISMATCH",
        "v1r2r1_runtime_target_created": False,
        "v1r2r1_artifact_count": 0,
        "v1r2r1_manual_retry_attempted": False,
        "v1r2r2_failure_preserved": True,
        "v1r2r2_cpu_create_invocation_count": 1,
        "v1r2r2_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2r2_exact_error_code": "PERMISSION_ERROR_WINERROR_5_RUNTIME_ROOT_CREATE",
        "v1r2r2_runtime_target_created": False,
        "v1r2r2_artifact_count": 0,
        "v1r2r2_manual_retry_attempted": False,
        "v1r2r3_failure_preserved": True,
        "v1r2r3_cpu_create_invocation_count": 1,
        "v1r2r3_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2r3_exact_error_code": "RUNNER_SETUP_PRISTINE_GATE",
        "v1r2r3_runtime_target_created": True,
        "v1r2r3_artifact_count": 7,
        "v1r2r3_cpu_artifacts_created": True,
        "v1r2r3_cpu_verification_passed": False,
        "v1r2r3_manual_retry_attempted": False,
        "scientific_artifacts_byte_equal_to_v1r1": True,
        "old_90_worker_evidence_preserved_read_only": True,
        "old_worker_rerun_allowed": False,
        "gpu_setup_probe_started": False,
        "gpu_numeric_smoke_started": False,
        "real_30_worker_started": False,
        "official_test_accessed": False,
        "reserve_1_accessed": False,
        "reserve_2_accessed": False,
        "test_dataset_access_count": 0,
    }
    for key, value in preservation_exact.items():
        require(preservation.get(key) == value, f"PRESERVATION_EXACT:{key}", gates)

    protocol = stored["gpu_execution_protocol.json"]
    require(protocol == runtime.build_gpu_execution_protocol(), "GPU_PROTOCOL_REPRODUCES", gates)
    require(protocol.get("status") == "PASS", "GPU_PROTOCOL_PASS", gates)
    require(protocol.get("condition_id") == "P2_EQ4", "GPU_PROTOCOL_P2_EQ4", gates)
    require(protocol.get("display_label") == "P2", "GPU_PROTOCOL_DISPLAY_P2", gates)
    require(protocol.get("separate_authorization_required") is True, "GPU_PROTOCOL_SEPARATE_AUTH", gates)
    require(protocol.get("mode_order") == ["GPU_SETUP_PROBE", "GPU_NUMERIC_SMOKE", "REAL_30_WORKER_P2_EQ4"], "GPU_PROTOCOL_MODE_ORDER", gates)
    require(protocol.get("automatic_retry") is False, "GPU_PROTOCOL_NO_RETRY", gates)
    require(protocol.get("test_and_reserve_contract", {}).get("official_test_access_allowed") is False, "GPU_PROTOCOL_NO_TEST", gates)
    require(protocol.get("test_and_reserve_contract", {}).get("test_dataset_access_count") == 0, "GPU_PROTOCOL_ZERO_TEST_ACCESS", gates)

    scientific = stored["scientific_protocol.json"]
    require(scientific.get("condition", {}).get("machine_id") == "P2_EQ4", "SCIENTIFIC_P2_EQ4", gates)
    require(scientific.get("condition", {}).get("layer_weights") == [0.25] * 4, "SCIENTIFIC_FIXED_QUARTERS", gates)
    require(scientific.get("condition", {}).get("layer_weights_trainable") is False, "SCIENTIFIC_WEIGHTS_FROZEN", gates)
    require(scientific.get("expected_parameter_contract") == EXPECTED_PARAMETER_CONTRACT, "SCIENTIFIC_PARAMETER_CONTRACT", gates)
    worker_matrix = stored["worker_matrix.json"]
    require(worker_matrix.get("worker_count") == 30, "WORKER_COUNT_30", gates)
    require(worker_matrix.get("old_worker_count_reused") == 90, "OLD_WORKER_COUNT_90", gates)
    require(worker_matrix.get("expected_optimizer_steps_per_worker") == 564, "WORKER_STEPS_564", gates)
    require([row.get("run_index") for row in worker_matrix.get("workers", [])] == list(range(1, 31)), "WORKER_INDICES_1_TO_30", gates)

    dry = stored["dry_gate_report.json"]
    dry_exact = {
        "status": "PASS",
        "runtime_version": "hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4",
        "preparation_mode": "CPU_ONLY",
        "condition_id": "P2_EQ4",
        "display_label": "P2",
        "new_worker_count": 30,
        "frozen_prior_worker_count": 90,
        "combined_analysis_worker_count": 120,
        "gpu_setup_probe_status": "NOT_STARTED",
        "gpu_numeric_smoke_status": "NOT_STARTED",
        "real_30_worker_status": "NOT_STARTED",
        "default_runner_status": "BLOCKED",
        "default_runner_exit_code": 3,
        "next_required_action": "SEPARATE_EXPLICIT_GPU_SETUP_PROBE_AUTHORIZATION",
        "no_existing_artifact_overwritten": True,
        "no_git_write": True,
        "automatic_retry": False,
        "model_load_started": False,
        "training_started": False,
        "forward_started": False,
        "backward_started": False,
        "prediction_started": False,
        "actual_optimizer_steps": 0,
        "scheduler_steps": 0,
        "official_test_accessed": False,
        "reserve_1_accessed": False,
        "reserve_2_accessed": False,
        "test_dataset_access_count": 0,
    }
    for key, value in dry_exact.items():
        require(dry.get(key) == value, f"DRY_EXACT:{key}", gates)
    actual_new_hashes = {
        "runtime": sha256_file(runtime_source),
        "verifier": sha256_file(verifier_source),
        "runner": sha256_file(runner),
    }
    require(dry.get("new_file_sha256") == actual_new_hashes, "DRY_NEW_FILE_HASHES", gates)
    prepared_hashes = {
        name: sha256_file(artifact_paths[name]) for name in CPU_ARTIFACT_NAMES[:-1]
    }
    require(dry.get("prepared_artifact_sha256") == prepared_hashes, "DRY_PREPARED_ARTIFACT_HASHES", gates)

    source = runtime_source.read_text(encoding="utf-8")
    tree = ast.parse(source)
    require("import hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1 as v1r1" in source, "RUNTIME_IMPORTS_V1R1", gates)
    require("import hpo_phi3_pooling_fixed4_followup_5x2cv_v1 as v1" in source, "RUNTIME_IMPORTS_V1", gates)
    for token in (
        "--prepare-cpu-only",
        "--confirm-gpu-setup-probe-v1r2r4",
        "--confirm-gpu-numeric-smoke-v1r2r4",
        "--confirm-real-30-worker-p2-eq4-v1r2r4",
        "--worker",
    ):
        require(token in source, f"RUNTIME_CLI_PRESENT:{token}", gates)
    lowered = source.lower()
    for forbidden in (
        "data/processed/test",
        "test_0.csv",
        "reserve1.json",
        "reserve2.json",
        "git add",
        "git commit",
        "git push",
    ):
        require(forbidden not in lowered, f"RUNTIME_FORBIDDEN_LITERAL_ABSENT:{forbidden}", gates)

    setup_node = named_function(tree, "run_gpu_setup_probe")
    setup_calls = call_names(setup_node)
    require(not any(name.endswith((".train", ".predict", ".backward", ".step")) for name in setup_calls), "SETUP_NO_TRAIN_PREDICT_BACKWARD_STEP", gates)
    smoke_node = named_function(tree, "run_gpu_numeric_smoke")
    smoke_calls = call_names(smoke_node)
    require(any(name.endswith(".backward") for name in smoke_calls), "SMOKE_BACKWARD_REQUIRED", gates)
    require(not any(name.endswith((".train", ".predict", ".step")) for name in smoke_calls), "SMOKE_NO_TRAIN_PREDICT_OR_STEP", gates)

    runner_text = runner.read_text(encoding="utf-8-sig")
    for switch in (
        "PrepareCpuOnly",
        "VerifyCpuOnly",
        "ConfirmGpuSetupProbeV1r2r4",
        "ConfirmGpuNumericSmokeV1r2r4",
        "ConfirmReal30WorkerP2Eq4V1r2r4",
    ):
        require(switch in runner_text, f"RUNNER_SWITCH:{switch}", gates)
    require(expected_runtime_sha256 in runner_text, "RUNNER_LITERAL_RUNTIME_HASH", gates)
    require(expected_verifier_sha256 in runner_text, "RUNNER_LITERAL_VERIFIER_HASH", gates)
    require(PACKAGE_HELPER_SHA256 in runner_text, "RUNNER_LITERAL_HELPER_HASH", gates)
    require("EXPLICIT_SINGLE_V1R2_MODE_REQUIRED" in runner_text, "RUNNER_SINGLE_MODE_GATE", gates)
    require("EXPLICIT_GPU_SETUP_PROBE_AUTHORIZATION_REQUIRED" in runner_text, "RUNNER_DEFAULT_GPU_BLOCK", gates)
    require("PYTHONNOUSERSITE" in runner_text and "PYTHONHASHSEED" in runner_text, "RUNNER_ISOLATED_PYTHON_ENV", gates)
    require("PACKAGE_ORIGIN_PREFLIGHT_FAILED" in runner_text, "RUNNER_PACKAGE_PREFLIGHT", gates)
    require("CPU_PACKAGE_VALIDATION_FAILED" in runner_text, "RUNNER_CPU_VALIDATION", gates)
    require(
        "--required-state $RequiredState" in runner_text
        and "Invoke-CpuVerifier 'cpu-pristine'" in runner_text,
        "RUNNER_SETUP_PRISTINE_GATE",
        gates,
    )
    require("Invoke-CpuVerifier 'setup-pass'" in runner_text, "RUNNER_SMOKE_SETUP_GATE", gates)
    require("Invoke-CpuVerifier 'smoke-pass'" in runner_text, "RUNNER_REAL_SMOKE_GATE", gates)
    require("SETUP_PROBE_TARGET_ALREADY_EXISTS" in runner_text, "RUNNER_SETUP_EXCLUSIVE_CREATE", gates)
    require("NUMERIC_SMOKE_TARGET_ALREADY_EXISTS" in runner_text, "RUNNER_SMOKE_EXCLUSIVE_CREATE", gates)
    require("REAL_30_RUNTIME_ARTIFACT_ALREADY_EXISTS" in runner_text, "RUNNER_REAL_EXCLUSIVE_CREATE", gates)
    require("NO_AUTOMATIC_RETRY=true" in runner_text, "RUNNER_NO_RETRY_STATUS", gates)
    inline_python = re.search(
        r"\$Python\b[^\r\n]*\s-c(?:\s|$)", runner_text, flags=re.IGNORECASE
    )
    require(inline_python is None and "@'" not in runner_text and '@"' not in runner_text, "RUNNER_NO_INLINE_PYTHON_OR_HERESTRING", gates)
    require("git add" not in runner_text.lower() and "git commit" not in runner_text.lower() and "git push" not in runner_text.lower(), "RUNNER_NO_GIT_WRITE", gates)

    setup_report = runtime_root / "gpu_setup_probe" / "setup_probe_report.json"
    smoke_report = runtime_root / "gpu_numeric_smoke" / "numeric_smoke_report.json"
    final_report = runtime_root / "p2_eq4_followup_report.json"
    incomplete_report = runtime_root / "p2_eq4_followup_incomplete_report.json"
    workers_root = runtime_root / "workers"
    dry_hash = sha256_file(artifact_paths["dry_gate_report.json"])
    runtime_hash = sha256_file(runtime_source)
    verifier_hash = sha256_file(verifier_source)

    if required_state == "cpu-pristine":
        for path in (setup_report.parent, smoke_report.parent, workers_root, final_report, incomplete_report):
            require(not path.exists(), f"CPU_PRISTINE_ABSENT:{path.name}", gates)
    elif required_state == "setup-pass":
        validate_setup_report(setup_report, runtime_hash, verifier_hash, dry_hash, gates)
        for path in (smoke_report.parent, workers_root, final_report, incomplete_report):
            require(not path.exists(), f"PRE_SMOKE_ABSENT:{path.name}", gates)
    elif required_state == "smoke-pass":
        validate_setup_report(setup_report, runtime_hash, verifier_hash, dry_hash, gates)
        validate_numeric_smoke_report(smoke_report, runtime_hash, verifier_hash, dry_hash, gates)
        for path in (workers_root, final_report, incomplete_report):
            require(not path.exists(), f"PRE_REAL_ABSENT:{path.name}", gates)
    elif required_state == "real-pass":
        validate_setup_report(setup_report, runtime_hash, verifier_hash, dry_hash, gates)
        validate_numeric_smoke_report(smoke_report, runtime_hash, verifier_hash, dry_hash, gates)
        require(not incomplete_report.exists(), "REAL_INCOMPLETE_ABSENT", gates)
        validate_real_report(final_report, workers_root, runtime_hash, verifier_hash, dry_hash, gates)
    elif required_state == "base-only":
        if setup_report.is_file():
            validate_setup_report(setup_report, runtime_hash, verifier_hash, dry_hash, gates)
        if smoke_report.is_file():
            validate_numeric_smoke_report(smoke_report, runtime_hash, verifier_hash, dry_hash, gates)
    else:
        raise VerificationError("REQUIRED_STATE_INVALID")

    return {
        "schema_version": "phi3_pooling_fixed4_followup_cpu_verification_v1r2",
        "status": "PASS",
        "required_state": required_state,
        "gate_count": len(gates),
        "gates": gates,
        "file_sha256": {
            "runtime": runtime_hash,
            "verifier": verifier_hash,
            "runner": sha256_file(runner),
            "dry_gate": dry_hash,
            "package_origin_helper": sha256_file(package_helper),
        },
        "v1r1_preserved": True,
        "scientific_artifacts_byte_equal_to_v1r1": True,
        "frozen_prior_worker_count": 90,
        "new_worker_count": 30,
        "security": {
            "verifier_cpu_only": True,
            "model_load_started_by_verifier": False,
            "cuda_initialized_by_verifier": False,
            "training_started_by_verifier": False,
            "forward_started_by_verifier": False,
            "backward_started_by_verifier": False,
            "prediction_started_by_verifier": False,
            "official_test_accessed_by_verifier": False,
            "automatic_retry": False,
            "no_git_write": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--runtime-source", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--required-state",
        choices=("base-only", "cpu-pristine", "setup-pass", "smoke-pass", "real-pass"),
        default="base-only",
    )
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        report = verify(
            project_root=arguments.project_root.resolve(),
            runtime_source=arguments.runtime_source.resolve(),
            runner=arguments.runner.resolve(),
            runtime_root=arguments.runtime_root.resolve(),
            required_state=arguments.required_state,
            expected_runtime_sha256=arguments.expected_runtime_sha256,
            expected_verifier_sha256=arguments.expected_verifier_sha256,
        )
    except Exception as error:
        code = str(error) if str(error) else type(error).__name__
        print(json.dumps({"status": "FAIL_CLOSED", "error_code": code}, sort_keys=True))
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
