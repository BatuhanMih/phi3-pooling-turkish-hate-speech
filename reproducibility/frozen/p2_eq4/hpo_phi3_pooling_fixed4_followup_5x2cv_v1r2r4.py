"""Immutable staged GPU execution package for the P2_EQ4 follow-up.

The CPU preparation reproduces the accepted V1r1 scientific artifacts and
adds only execution/provenance contracts.  GPU actions are mutually exclusive
and require separate explicit invocations: setup probe, one-minibatch numeric
smoke, and the real 30-worker P2_EQ4 follow-up.  No path or mode in this module
can access an official test or reserve dataset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import hpo_phi3_pooling_confirmatory_5x2cv_v1 as prior
import hpo_phi3_pooling_fixed4_followup_5x2cv_v1 as v1
import hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1 as v1r1
import hpo_phi3_pooling_head_screen_v1r2 as pooling


RUNTIME_VERSION: Final = "hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4"
RUNTIME_RELATIVE: Final = "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.py"
VERIFIER_RELATIVE: Final = "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.py"
RUNNER_RELATIVE: Final = "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4.ps1"
RUNTIME_ROOT_RELATIVE: Final = "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r4"
HPO_ROOT_RELATIVE: Final = "outputs/hpo_artifacts_v1"
FIXED_PYTHON: Final = v1.FIXED_PYTHON

CONDITION_ID: Final = "P2_EQ4"
DISPLAY_LABEL: Final = "P2"
EXPECTED_WORKERS: Final = 30
EXPECTED_OPTIMIZER_STEPS: Final = 564
T_CRITICAL_95_DF4: Final = 2.7764451051977987
EXPECTED_RUNTIME_VERSIONS: Final = {
    "python_major": 3,
    "python_minor": 11,
    "torch_version": "2.5.1+cu121",
    "transformers_version": "4.42.4",
    "peft_version": "0.11.1",
}
EXPECTED_PARAMETER_CONTRACT: Final = {
    "total_parameter_count": 1_916_943_360,
    "trainable_parameter_count": 6_297_600,
    "trainable_parameter_groups": {
        "attention_pool": 0,
        "classifier": 6_144,
        "layer_mix": 0,
        "lora": 6_291_456,
    },
}

V1R1_PRESERVATION_SHA256: Final[dict[str, str]] = {
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1.py": "66fd56123dc0294f7296b9b70b35b9e0f9089faeecf517898a452a1ca78bb21e",
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1.py": "f1e7e8a8e5ea4c0100226494a5e583bcf87278642651a1fcb8fb4893fe987523",
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1.ps1": "7ba0f750d12ab8cbd9d52df8014d9a23090b0d6bcd23bf4c49070f1e1b5a8ce9",
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1.py": "cd6f1720de95a42f7e74f5893303aa262bd42544d16db702f7b8ddf4871a8d81",
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1.py": "c7b86ab3392a6858afb40159805723652075517cd39a00c8e173eafdd898249e",
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1.ps1": "fda4c65c23afe9bb165a0bbbd50c212240ad5d7af2588bd1d7c9c8f1c8f73caf",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/cpu_equivalence_report.json": "1b7adb31a80da5f69738b7b8959b65bc141e5d758f61c722790a771326cfbe57",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/dry_gate_report.json": "9c031bd9a01fce5a1e11bb8be72fe8a2cb9af32765881523453cd09f278bd4a1",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/scientific_protocol.json": "8899452939cff2626c4f234e52af1729a477caab5cc5cb38ca8295ea74bde85b",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/source_manifest.json": "6c05382301242fe563f8c980dfc1c4184a6c54bbe3b63015d02a704bef9bbace",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/v1_failure_preservation_report.json": "4e9f6e589903c2b72113f91c6123a4b6c72a95f43487a1d6e3b72731407c05de",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r1/worker_matrix.json": "3e5a8e8b02cca00f0e9727fea2d25604196bcc96201061eacd8b59e5813bb102",
}

V1R2_FAILURE_PRESERVATION_SHA256: Final[dict[str, str]] = {
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2.py": "d274b08c01509fa3c2bd4f1ad5375d956a42d852738aaeec2de098a026eb7532",
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2.py": "89dfe9ac048e3a2c1b54e0eb70bb93ea9b730760645b125f2b1d25da749ef7f0",
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2.ps1": "88225a07a1275cbbfe0aa4397f0a6b3deb6b2f1e6997b8f049ea99fef5d02903",
}

V1R2R1_FAILURE_PRESERVATION_SHA256: Final[dict[str, str]] = {
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r1.py": "9d8a9394dea81898c9dacb959464e2c50fe9a69e26ae0999d89605226d36532e",
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r1.py": "90f384134e0d1e148129b65979f8312a7e4731911562a544283de29b6a4fb929",
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r1.ps1": "4f4ad850a3f69c8a2dd9ab81bebf2ae4dc24e255b3b3aa82b3a87b663658f2f6",
}

V1R2R2_FAILURE_PRESERVATION_SHA256: Final[dict[str, str]] = {
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r2.py": "f2301086e95b7504a9e52d505a935a243992a816a58a59d166d6aefebbda21e6",
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r2.py": "de4357481b3e5bd5e06bbefe54eb9e9c48b78778ea0d9337987a9e4d0efea0af",
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r2.ps1": "45b3967203fa9becc84db16203d19de87f1fe0c58adc49985ff79113d42773f3",
}

V1R2R3_FAILURE_PRESERVATION_SHA256: Final[dict[str, str]] = {
    "src/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3.py": "3b5fd03a786a2f32335ad68f019c53d22d0fb86b5b1c71272021c20afe55e6e2",
    "tools/verify_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3.py": "f9355e8a8882cf4135908d0de25d0cff2c41664b3e1b6421417f46d0abc35ee4",
    "run_hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3.ps1": "9fda37b360f3951aa6de03abba6a000fc761b69f186a22a1577678df88cd06ef",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/cpu_equivalence_report.json": "1b7adb31a80da5f69738b7b8959b65bc141e5d758f61c722790a771326cfbe57",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/dry_gate_report.json": "70849c78ba1361234caa286bb6e53c9d7143ba3d6c905f76e7c095e1a7e47572",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/gpu_execution_protocol.json": "48c1355a59d3af5db352bb45fbb383d88688bb273991d900fa9b30eaca633728",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/scientific_protocol.json": "8899452939cff2626c4f234e52af1729a477caab5cc5cb38ca8295ea74bde85b",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/source_manifest.json": "6c05382301242fe563f8c980dfc1c4184a6c54bbe3b63015d02a704bef9bbace",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/v1r1_preservation_report.json": "76221475f809a415564ac94f6547c4f4c394cc05c2a3a8adfe12230c3025eb6b",
    "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3/worker_matrix.json": "3e5a8e8b02cca00f0e9727fea2d25604196bcc96201061eacd8b59e5813bb102",
}

SCIENTIFIC_ARTIFACT_NAMES: Final = (
    "source_manifest.json",
    "scientific_protocol.json",
    "worker_matrix.json",
    "cpu_equivalence_report.json",
)


class FollowupV1r2Error(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    project_root: Path
    hpo_artifact_root: Path
    runtime_root: Path


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FollowupV1r2Error("LOCKED_FILE_MISSING_OR_SYMLINK")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise FollowupV1r2Error(code) from None
    if not isinstance(value, dict):
        raise FollowupV1r2Error(code)
    return value


def create_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    data = canonical_json_bytes(dict(value))
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise FollowupV1r2Error("EXCLUSIVE_ARTIFACT_CREATE_FAILED") from None


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def locked_paths(project_root: Path) -> RuntimePaths:
    canonical_project = Path(__file__).resolve().parents[1]
    if not _same_path(project_root, canonical_project):
        raise FollowupV1r2Error("PROJECT_ROOT_CONTRACT_MISMATCH")
    return RuntimePaths(
        project_root=canonical_project,
        hpo_artifact_root=(canonical_project / HPO_ROOT_RELATIVE).resolve(),
        runtime_root=(canonical_project / RUNTIME_ROOT_RELATIVE).resolve(),
    )


def verify_runtime_paths(paths: RuntimePaths) -> None:
    expected = locked_paths(paths.project_root)
    if not (
        _same_path(paths.project_root, expected.project_root)
        and _same_path(paths.hpo_artifact_root, expected.hpo_artifact_root)
        and _same_path(paths.runtime_root, expected.runtime_root)
    ):
        raise FollowupV1r2Error("RUNTIME_PATH_CONTRACT_MISMATCH")
    if not paths.hpo_artifact_root.is_dir() or paths.hpo_artifact_root.is_symlink():
        raise FollowupV1r2Error("HPO_ARTIFACT_ROOT_INVALID")


def verify_fixed_interpreter() -> None:
    if not _same_path(Path(sys.executable), Path(FIXED_PYTHON)):
        raise FollowupV1r2Error("FIXED_INTERPRETER_MISMATCH")


def verify_v1r1_preserved(project_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V1R1_PRESERVATION_SHA256.items():
        actual = sha256_file(project_root / relative)
        if actual != expected:
            raise FollowupV1r2Error(f"V1R1_PRESERVATION_HASH_MISMATCH:{relative}")
        observed[relative] = actual
    # Re-hash all 180 immutable old result/status files through the accepted V1 reader.
    if len(v1.collect_old_worker_artifacts(project_root)) != 90:
        raise FollowupV1r2Error("OLD_90_WORKER_EVIDENCE_INVALID")
    return observed


def verify_v1r2_failure_preserved(project_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V1R2_FAILURE_PRESERVATION_SHA256.items():
        actual = sha256_file(project_root / relative)
        if actual != expected:
            raise FollowupV1r2Error(f"V1R2_FAILURE_PRESERVATION_HASH_MISMATCH:{relative}")
        observed[relative] = actual
    if (project_root / "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2").exists():
        raise FollowupV1r2Error("V1R2_FAILED_TARGET_MUST_REMAIN_ABSENT")
    return observed


def verify_v1r2r1_failure_preserved(project_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V1R2R1_FAILURE_PRESERVATION_SHA256.items():
        actual = sha256_file(project_root / relative)
        if actual != expected:
            raise FollowupV1r2Error(f"V1R2R1_FAILURE_PRESERVATION_HASH_MISMATCH:{relative}")
        observed[relative] = actual
    if (project_root / "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r1").exists():
        raise FollowupV1r2Error("V1R2R1_FAILED_TARGET_MUST_REMAIN_ABSENT")
    return observed


def verify_v1r2r2_failure_preserved(project_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V1R2R2_FAILURE_PRESERVATION_SHA256.items():
        actual = sha256_file(project_root / relative)
        if actual != expected:
            raise FollowupV1r2Error(f"V1R2R2_FAILURE_PRESERVATION_HASH_MISMATCH:{relative}")
        observed[relative] = actual
    if (project_root / "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r2").exists():
        raise FollowupV1r2Error("V1R2R2_FAILED_TARGET_MUST_REMAIN_ABSENT")
    return observed


def verify_v1r2r3_failure_preserved(project_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V1R2R3_FAILURE_PRESERVATION_SHA256.items():
        actual = sha256_file(project_root / relative)
        if actual != expected:
            raise FollowupV1r2Error(f"V1R2R3_FAILURE_PRESERVATION_HASH_MISMATCH:{relative}")
        observed[relative] = actual
    failed_root = project_root / "runtime/hpo_phi3_pooling_fixed4_followup_5x2cv_v1r2r3"
    expected_names = {
        "cpu_equivalence_report.json",
        "dry_gate_report.json",
        "gpu_execution_protocol.json",
        "scientific_protocol.json",
        "source_manifest.json",
        "v1r1_preservation_report.json",
        "worker_matrix.json",
    }
    if not failed_root.is_dir() or failed_root.is_symlink():
        raise FollowupV1r2Error("V1R2R3_FAILED_TARGET_MUST_REMAIN_PRESENT")
    if {path.name for path in failed_root.iterdir()} != expected_names:
        raise FollowupV1r2Error("V1R2R3_FAILED_TARGET_CONTENT_MISMATCH")
    return observed


def _locked_scientific_payloads(project_root: Path) -> dict[str, dict[str, Any]]:
    root = project_root / v1r1.RUNTIME_ROOT_RELATIVE
    payloads = {
        name: load_json(root / name, f"V1R1_ARTIFACT_INVALID:{name}")
        for name in SCIENTIFIC_ARTIFACT_NAMES
    }
    recomputed = v1r1.scientific_payloads(project_root)
    if payloads != recomputed:
        raise FollowupV1r2Error("V1R1_SCIENTIFIC_PAYLOAD_REPRODUCTION_MISMATCH")
    return payloads


def build_v1r1_preservation_report(project_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "phi3_pooling_fixed4_followup_revision_preservation_v1r2r4",
        "status": "PASS",
        "v1r1_preserved": True,
        "v1r1_preserved_sha256": verify_v1r1_preserved(project_root),
        "v1r2_failure_preserved": True,
        "v1r2_failure_preserved_sha256": verify_v1r2_failure_preserved(project_root),
        "v1r2_cpu_create_invocation_count": 1,
        "v1r2_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2_exact_error_code": "PERMISSION_ERROR_WINERROR_5_RUNTIME_ROOT_CREATE",
        "v1r2_root_cause": "SANDBOX_WRITE_PERMISSION_NOT_GRANTED_TO_CHILD_PROCESS",
        "v1r2_package_origin_preflight": "7_OF_7_PASS",
        "v1r2_runtime_target_created": False,
        "v1r2_artifact_count": 0,
        "v1r2_manual_retry_attempted": False,
        "v1r2r1_failure_preserved": True,
        "v1r2r1_failure_preserved_sha256": verify_v1r2r1_failure_preserved(project_root),
        "v1r2r1_cpu_create_invocation_count": 1,
        "v1r2r1_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2r1_exact_error_code": "ARGPARSE_NAMESPACE_ATTRIBUTE_REVISION_MISMATCH",
        "v1r2r1_root_cause": "MECHANICAL_REVISION_LEFT_INTERNAL_ARGPARSE_ATTRIBUTE_NAMES_AT_V1R2",
        "v1r2r1_package_origin_preflight": "7_OF_7_PASS",
        "v1r2r1_runtime_target_created": False,
        "v1r2r1_artifact_count": 0,
        "v1r2r1_manual_retry_attempted": False,
        "v1r2r2_failure_preserved": True,
        "v1r2r2_failure_preserved_sha256": verify_v1r2r2_failure_preserved(project_root),
        "v1r2r2_cpu_create_invocation_count": 1,
        "v1r2r2_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2r2_exact_error_code": "PERMISSION_ERROR_WINERROR_5_RUNTIME_ROOT_CREATE",
        "v1r2r2_root_cause": "SANDBOX_CHILD_FILESYSTEM_PERMISSION_DENIED",
        "v1r2r2_package_origin_preflight": "7_OF_7_PASS",
        "v1r2r2_runtime_target_created": False,
        "v1r2r2_artifact_count": 0,
        "v1r2r2_manual_retry_attempted": False,
        "v1r2r3_failure_preserved": True,
        "v1r2r3_failure_preserved_sha256": verify_v1r2r3_failure_preserved(project_root),
        "v1r2r3_cpu_create_invocation_count": 1,
        "v1r2r3_cpu_create_status": "INCOMPLETE_FAIL_CLOSED",
        "v1r2r3_exact_error_code": "RUNNER_SETUP_PRISTINE_GATE",
        "v1r2r3_root_cause": "VERIFIER_LITERAL_STATE_CHECK_AND_RUNNER_SWITCH_REVISION_CASE_MISMATCH",
        "v1r2r3_package_origin_preflight": "7_OF_7_PASS",
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
        "automatic_retry_attempted": False,
        "existing_artifact_overwritten": False,
        "git_write_performed": False,
    }


def build_gpu_execution_protocol() -> dict[str, Any]:
    return {
        "schema_version": "phi3_pooling_fixed4_followup_gpu_execution_protocol_v1r2r4",
        "status": "PASS",
        "runtime_version": RUNTIME_VERSION,
        "condition_id": CONDITION_ID,
        "display_label": DISPLAY_LABEL,
        "condition_description": "fixed_equal_last_four_hidden_layers_then_masked_mean",
        "mode_order": [
            "GPU_SETUP_PROBE",
            "GPU_NUMERIC_SMOKE",
            "REAL_30_WORKER_P2_EQ4",
        ],
        "separate_authorization_required": True,
        "automatic_retry": False,
        "resume_allowed": False,
        "setup_probe": {
            "stop_stage": "trainer_created",
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "prediction_calls": 0,
        },
        "numeric_smoke": {
            "data_scope": "one_prelocked_development_training_minibatch",
            "loss_path": "transformers.Trainer.compute_loss_with_label_smoothing_0.05",
            "forward_calls": 1,
            "backward_calls": 1,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "prediction_calls": 0,
            "checkpoint_writes": 0,
        },
        "real_execution": {
            "worker_count": EXPECTED_WORKERS,
            "worker_processes_sequential": True,
            "fresh_process_per_worker": True,
            "expected_optimizer_steps_per_worker": EXPECTED_OPTIMIZER_STEPS,
            "automatic_retry": False,
            "selection_or_retuning": False,
        },
        "analysis": {
            "unit": "repetition",
            "repetition_count": 5,
            "split_score": "mean_of_three_training_replicates",
            "repetition_score": "mean_of_two_directions",
            "primary_follow_up_contrast": "P5_minus_P2_EQ4",
            "secondary_follow_up_contrast": "P2_EQ4_minus_P1",
            "context_contrasts": ["P1_minus_P0", "P5_minus_P0"],
            "interval": "two_sided_paired_student_t_95_percent_df_4",
            "binary_significance_or_selection_gate": False,
            "causal_or_contemporaneous_claim_allowed": False,
        },
        "residual_limit": (
            "The old P0/P1/P5 workers and new P2_EQ4 workers are temporally "
            "separated. Hash, environment, fold, seed, order, budget and formula "
            "gates reduce but cannot eliminate temporal CUDA nondeterminism."
        ),
        "test_and_reserve_contract": {
            "official_test_access_allowed": False,
            "reserve_1_access_allowed": False,
            "reserve_2_access_allowed": False,
            "test_dataset_access_count": 0,
        },
    }


def build_cpu_payloads(project_root: Path) -> dict[str, dict[str, Any]]:
    scientific = _locked_scientific_payloads(project_root)
    return {
        **scientific,
        "v1r1_preservation_report.json": build_v1r1_preservation_report(project_root),
        "gpu_execution_protocol.json": build_gpu_execution_protocol(),
    }


def _new_file_hashes(project_root: Path) -> dict[str, str]:
    return {
        "runtime": sha256_file(project_root / RUNTIME_RELATIVE),
        "verifier": sha256_file(project_root / VERIFIER_RELATIVE),
        "runner": sha256_file(project_root / RUNNER_RELATIVE),
    }


def build_dry_gate(
    project_root: Path, payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    prepared_hashes = {
        name: hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        for name, value in payloads.items()
    }
    return {
        "schema_version": "phi3_pooling_fixed4_followup_dry_gate_v1r2r4",
        "status": "PASS",
        "runtime_version": RUNTIME_VERSION,
        "preparation_mode": "CPU_ONLY",
        "implementation_reuse": "IMMUTABLE_V1_FORMULA_AND_V1R1_SCIENTIFIC_ARTIFACTS",
        "scientific_formula_changed": False,
        "worker_plan_changed": False,
        "condition_id": CONDITION_ID,
        "display_label": DISPLAY_LABEL,
        "new_worker_count": EXPECTED_WORKERS,
        "frozen_prior_worker_count": 90,
        "combined_analysis_worker_count": 120,
        "new_file_sha256": _new_file_hashes(project_root),
        "prepared_artifact_sha256": prepared_hashes,
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


def prepare_cpu_only(project_root: Path) -> dict[str, Any]:
    verify_fixed_interpreter()
    paths = locked_paths(project_root)
    verify_runtime_paths(paths)
    if paths.runtime_root.exists():
        raise FollowupV1r2Error("V1R2_RUNTIME_ROOT_ALREADY_EXISTS")
    payloads = build_cpu_payloads(paths.project_root)
    dry = build_dry_gate(paths.project_root, payloads)
    paths.runtime_root.mkdir(parents=False, exist_ok=False)
    for name, value in payloads.items():
        create_json_exclusive(paths.runtime_root / name, value)
    create_json_exclusive(paths.runtime_root / "dry_gate_report.json", dry)
    return {
        "status": "PASS",
        "mode": "CPU_ONLY_PREPARATION",
        "artifact_count": 7,
        "runtime_root": str(paths.runtime_root),
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in sorted(paths.runtime_root.iterdir(), key=lambda item: item.name)
        },
        "v1r1_preserved": True,
        "scientific_artifacts_byte_equal_to_v1r1": True,
        "new_workers_planned": EXPECTED_WORKERS,
        "gpu_setup_probe_status": "NOT_STARTED",
        "gpu_numeric_smoke_status": "NOT_STARTED",
        "real_30_worker_status": "NOT_STARTED",
    }


def _cpu_artifact_names() -> tuple[str, ...]:
    return (
        "source_manifest.json",
        "scientific_protocol.json",
        "worker_matrix.json",
        "cpu_equivalence_report.json",
        "v1r1_preservation_report.json",
        "gpu_execution_protocol.json",
        "dry_gate_report.json",
    )


def verify_cpu_artifacts(paths: RuntimePaths) -> tuple[dict[str, Any], str]:
    verify_fixed_interpreter()
    verify_runtime_paths(paths)
    verify_v1r1_preserved(paths.project_root)
    expected_names = set(_cpu_artifact_names())
    if not paths.runtime_root.is_dir() or paths.runtime_root.is_symlink():
        raise FollowupV1r2Error("V1R2_RUNTIME_ROOT_INVALID")
    for name in expected_names:
        path = paths.runtime_root / name
        if not path.is_file() or path.is_symlink():
            raise FollowupV1r2Error(f"CPU_ARTIFACT_INVALID:{name}")
    payloads = build_cpu_payloads(paths.project_root)
    for name, expected in payloads.items():
        if load_json(paths.runtime_root / name, f"CPU_ARTIFACT_INVALID:{name}") != expected:
            raise FollowupV1r2Error(f"CPU_ARTIFACT_REPRODUCTION_MISMATCH:{name}")
    dry = load_json(paths.runtime_root / "dry_gate_report.json", "DRY_GATE_INVALID")
    if dry != build_dry_gate(paths.project_root, payloads):
        raise FollowupV1r2Error("DRY_GATE_REPRODUCTION_MISMATCH")
    return dry, sha256_file(paths.runtime_root / "dry_gate_report.json")


def _expected_report_links(paths: RuntimePaths, dry_hash: str) -> dict[str, str]:
    return {
        "runtime_source_sha256": sha256_file(paths.project_root / RUNTIME_RELATIVE),
        "cpu_verifier_sha256": sha256_file(paths.project_root / VERIFIER_RELATIVE),
        "dry_gate_report_sha256": dry_hash,
    }


def _verify_report_links(report: Mapping[str, Any], links: Mapping[str, str]) -> None:
    if any(report.get(key) != value for key, value in links.items()):
        raise FollowupV1r2Error("GPU_REPORT_HASH_LINK_MISMATCH")


def _verify_setup_report(paths: RuntimePaths, dry_hash: str) -> dict[str, Any]:
    report = load_json(
        paths.runtime_root / "gpu_setup_probe" / "setup_probe_report.json",
        "SETUP_REPORT_INVALID",
    )
    required = {
        "schema_version": "phi3_pooling_fixed4_followup_setup_probe_v1r2r4",
        "status": "PASS",
        "mode": "GPU_SETUP_PROBE",
        "setup_probe_status": "PASS",
        "condition_id": CONDITION_ID,
        "model_loaded": True,
        "trainer_created": True,
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
    if any(report.get(key) != value for key, value in required.items()):
        raise FollowupV1r2Error("SETUP_REPORT_CONTRACT_MISMATCH")
    if report.get("trainable_parameter_groups") != EXPECTED_PARAMETER_CONTRACT["trainable_parameter_groups"]:
        raise FollowupV1r2Error("SETUP_PARAMETER_CONTRACT_MISMATCH")
    _verify_report_links(report, _expected_report_links(paths, dry_hash))
    return report


def _verify_smoke_report(paths: RuntimePaths, dry_hash: str) -> dict[str, Any]:
    report = load_json(
        paths.runtime_root / "gpu_numeric_smoke" / "numeric_smoke_report.json",
        "NUMERIC_SMOKE_REPORT_INVALID",
    )
    required = {
        "schema_version": "phi3_pooling_fixed4_followup_numeric_smoke_v1r2r4",
        "status": "PASS",
        "mode": "GPU_NUMERIC_SMOKE",
        "numeric_smoke_status": "PASS",
        "condition_id": CONDITION_ID,
        "model_loaded": True,
        "trainer_created": True,
        "full_training_started": False,
        "forward_started": True,
        "backward_started": True,
        "prediction_started": False,
        "checkpoint_written": False,
        "actual_optimizer_steps": 0,
        "scheduler_steps": 0,
        "loss_finite": True,
        "logits_finite": True,
        "gradient_finite": True,
        "fixed_layer_weights_trainable": False,
        "official_test_accessed": False,
        "reserve_1_accessed": False,
        "reserve_2_accessed": False,
        "test_dataset_access_count": 0,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise FollowupV1r2Error("NUMERIC_SMOKE_REPORT_CONTRACT_MISMATCH")
    _verify_report_links(report, _expected_report_links(paths, dry_hash))
    return report


class LockedP2Classifier(v1.FixedEqualFourLayerClassifier):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self._v1r2_project_root = project_root

    def _load_heavy_runtime(self) -> Any:
        config = super()._load_heavy_runtime()
        observed = {
            "python_major": int(sys.version_info.major),
            "python_minor": int(sys.version_info.minor),
            "torch_version": str(self.torch.__version__),
            "transformers_version": str(self.transformers.__version__),
            "peft_version": str(self.peft.__version__),
        }
        if observed != EXPECTED_RUNTIME_VERSIONS:
            raise FollowupV1r2Error("RUNTIME_VERSION_CONTRACT_MISMATCH")
        site_packages = (self._v1r2_project_root / "qlora_env/Lib/site-packages").resolve()
        for module in (self.torch, self.transformers, self.peft, self.numpy):
            origin = Path(str(getattr(module, "__file__", ""))).resolve()
            try:
                origin.relative_to(site_packages)
            except Exception:
                raise FollowupV1r2Error("HEAVY_PACKAGE_ORIGIN_MISMATCH") from None
        return config


def _first_worker_context(paths: RuntimePaths) -> tuple[Any, Mapping[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    spec = v1.build_worker_specs()[0]
    plan = load_json(paths.project_root / v1.OLD_FOLD_PLAN_RELATIVE, "OLD_FOLD_PLAN_INVALID")
    records = prior.load_development_records(
        prior.RuntimePaths(paths.project_root, paths.hpo_artifact_root, paths.runtime_root)
    )
    prior.validate_fold_plan(plan, records)
    split = prior._plan_split(plan, spec.repetition, spec.direction)
    train_positions = tuple(int(item) for item in split["train_positions"])
    eval_positions = tuple(int(item) for item in split["ordered_eval_positions"])
    train_records = tuple(records[item] for item in train_positions)
    validation_records = tuple(records[item] for item in eval_positions)
    if len(train_records) != 750 or len(validation_records) != 750:
        raise FollowupV1r2Error("SETUP_SPLIT_SIZE_MISMATCH")
    return spec, split, train_records, validation_records


def _validate_setup_result(result: Mapping[str, Any]) -> None:
    if (
        result.get("setup_probe_status") != "PASS"
        or result.get("condition_id") != CONDITION_ID
        or result.get("actual_optimizer_steps") != 0
        or result.get("training_started") is not False
        or result.get("prediction_started") is not False
        or result.get("total_parameter_count") != EXPECTED_PARAMETER_CONTRACT["total_parameter_count"]
        or result.get("trainable_parameter_count") != EXPECTED_PARAMETER_CONTRACT["trainable_parameter_count"]
        or result.get("trainable_parameter_groups") != EXPECTED_PARAMETER_CONTRACT["trainable_parameter_groups"]
    ):
        raise FollowupV1r2Error("P2_SETUP_RESULT_CONTRACT_MISMATCH")


def _diagnostic_failure(
    *,
    error: BaseException,
    recorder: Any,
    root: Path,
    links: Mapping[str, str],
    classifier: Any,
) -> tuple[str, str, str]:
    diagnostic_error: BaseException = error
    if isinstance(error, FollowupV1r2Error):
        diagnostic_error = pooling.PoolingScreenError(error.code)
    code, path, digest = pooling.persist_diagnostic_exception(
        error=diagnostic_error,
        recorder=recorder,
        worker_root=root,
        runtime_sha256=links["runtime_source_sha256"],
        verifier_sha256=links["cpu_verifier_sha256"],
        dry_gate_sha256=links["dry_gate_report_sha256"],
        torch_module=getattr(classifier, "torch", None),
    )
    return code, str(path), digest


def run_gpu_setup_probe(paths: RuntimePaths) -> dict[str, Any]:
    dry, dry_hash = verify_cpu_artifacts(paths)
    del dry
    target = paths.runtime_root / "gpu_setup_probe"
    forbidden = (
        paths.runtime_root / "gpu_numeric_smoke",
        paths.runtime_root / "workers",
        paths.runtime_root / "p2_eq4_followup_report.json",
        paths.runtime_root / "p2_eq4_followup_incomplete_report.json",
    )
    if target.exists() or any(path.exists() for path in forbidden):
        raise FollowupV1r2Error("SETUP_PROBE_TARGET_ALREADY_EXISTS")
    target.mkdir(exist_ok=False)
    spec, split, train_records, validation_records = _first_worker_context(paths)
    recorder = pooling.DiagnosticStageRecorder(
        target, condition_id=CONDITION_ID, fold_index=spec.fold_index, training_seed=spec.training_seed
    )
    classifier = LockedP2Classifier(paths.project_root)
    links = _expected_report_links(paths, dry_hash)
    try:
        result = classifier.evaluate_pooling_diagnostic(
            recorder=recorder,
            setup_probe=True,
            condition=v1.p2_condition_spec(),
            fold_index=spec.fold_index,
            fold_membership_seed=spec.partition_seed,
            training_seed=spec.training_seed,
            train_records=train_records,
            validation_records=validation_records,
            eval_positions=tuple(int(item) for item in split["ordered_eval_positions"]),
            validation_order_provenance_id=str(split["validation_order_provenance_id"]),
        )
        if not isinstance(result, dict):
            raise FollowupV1r2Error("SETUP_RESULT_TYPE_MISMATCH")
        _validate_setup_result(result)
        report = {
            "schema_version": "phi3_pooling_fixed4_followup_setup_probe_v1r2r4",
            "status": "PASS",
            "mode": "GPU_SETUP_PROBE",
            "setup_probe_status": "PASS",
            "condition_id": CONDITION_ID,
            "model_id": pooling.MODEL_ID,
            "model_revision": pooling.MODEL_REVISION,
            "fold_index": spec.fold_index,
            "partition_seed": spec.partition_seed,
            "training_seed": spec.training_seed,
            "model_loaded": True,
            "trainer_created": True,
            "stop_stage": "trainer_created",
            "forward_started": False,
            "backward_started": False,
            "training_started": False,
            "prediction_started": False,
            "checkpoint_written": False,
            "actual_optimizer_steps": 0,
            "scheduler_steps": 0,
            "total_parameter_count": result["total_parameter_count"],
            "trainable_parameter_count": result["trainable_parameter_count"],
            "trainable_parameter_groups": result["trainable_parameter_groups"],
            "runtime_versions": dict(EXPECTED_RUNTIME_VERSIONS),
            "peak_vram_mib": result["peak_vram_mib"],
            "elapsed_seconds": result["elapsed_seconds"],
            **links,
            "official_test_accessed": False,
            "reserve_1_accessed": False,
            "reserve_2_accessed": False,
            "test_dataset_access_count": 0,
            "automatic_retry_attempted": False,
        }
        create_json_exclusive(target / "setup_probe_report.json", report)
        return report
    except BaseException as error:
        code, diagnostic_path, diagnostic_sha256 = _diagnostic_failure(
            error=error, recorder=recorder, root=target, links=links, classifier=classifier
        )
        if not (target / "setup_probe_report.json").exists():
            create_json_exclusive(
                target / "setup_probe_report.json",
                {
                    "schema_version": "phi3_pooling_fixed4_followup_setup_probe_v1r2r4",
                    "status": "INCOMPLETE_FAIL_CLOSED",
                    "mode": "GPU_SETUP_PROBE",
                    "setup_probe_status": "INCOMPLETE_FAIL_CLOSED",
                    "error_code": code,
                    "last_successful_stage": recorder.last_successful_stage,
                    "failed_stage": recorder.current_stage,
                    "diagnostic_exception_path": diagnostic_path,
                    "diagnostic_exception_sha256": diagnostic_sha256,
                    "actual_optimizer_steps": 0,
                    "scheduler_steps": 0,
                    "training_started": False,
                    "prediction_started": False,
                    **links,
                    "official_test_accessed": False,
                    "reserve_1_accessed": False,
                    "reserve_2_accessed": False,
                    "test_dataset_access_count": 0,
                    "automatic_retry_attempted": False,
                },
            )
        raise FollowupV1r2Error(code) from None
    finally:
        classifier.cleanup()
        gc.collect()


def _prepare_smoke_loss(classifier: LockedP2Classifier) -> tuple[Any, Any]:
    classifier.model.train()
    iterator = iter(classifier.trainer.get_train_dataloader())
    batch = next(iterator)
    prepared = classifier.trainer._prepare_inputs(batch)
    return classifier.trainer.compute_loss(
        classifier.model, prepared, return_outputs=True
    )


def _gradient_summary(classifier: LockedP2Classifier) -> dict[str, Any]:
    torch_module = classifier.torch
    groups: dict[str, dict[str, Any]] = {
        "lora": {"tensor_count": 0, "nonzero_absolute_sum": 0.0, "finite": True},
        "classifier": {"tensor_count": 0, "nonzero_absolute_sum": 0.0, "finite": True},
    }
    for name, parameter in classifier.model.named_parameters():
        if "layer_mix" in name.lower():
            raise FollowupV1r2Error("P2_FIXED_LAYER_MIX_PARAMETER_PRESENT")
        if parameter.grad is None:
            continue
        lowered = name.lower()
        group_name = "lora" if "lora_" in lowered else "classifier" if "score" in lowered else None
        if group_name is None:
            continue
        gradient = parameter.grad.detach()
        groups[group_name]["tensor_count"] += 1
        groups[group_name]["nonzero_absolute_sum"] += float(gradient.abs().sum().item())
        groups[group_name]["finite"] = bool(
            groups[group_name]["finite"] and torch_module.isfinite(gradient).all().item()
        )
    for group in groups.values():
        if group["tensor_count"] <= 0 or group["nonzero_absolute_sum"] <= 0.0 or group["finite"] is not True:
            raise FollowupV1r2Error("NUMERIC_SMOKE_GRADIENT_CONTRACT_MISMATCH")
    return groups


def run_gpu_numeric_smoke(paths: RuntimePaths) -> dict[str, Any]:
    _, dry_hash = verify_cpu_artifacts(paths)
    _verify_setup_report(paths, dry_hash)
    target = paths.runtime_root / "gpu_numeric_smoke"
    forbidden = (
        paths.runtime_root / "workers",
        paths.runtime_root / "p2_eq4_followup_report.json",
        paths.runtime_root / "p2_eq4_followup_incomplete_report.json",
    )
    if target.exists() or any(path.exists() for path in forbidden):
        raise FollowupV1r2Error("NUMERIC_SMOKE_TARGET_ALREADY_EXISTS")
    target.mkdir(exist_ok=False)
    spec, split, train_records, validation_records = _first_worker_context(paths)
    recorder = pooling.DiagnosticStageRecorder(
        target, condition_id=CONDITION_ID, fold_index=spec.fold_index, training_seed=spec.training_seed
    )
    classifier = LockedP2Classifier(paths.project_root)
    links = _expected_report_links(paths, dry_hash)
    started = time.perf_counter()
    try:
        setup = classifier.evaluate_pooling_diagnostic(
            recorder=recorder,
            setup_probe=True,
            condition=v1.p2_condition_spec(),
            fold_index=spec.fold_index,
            fold_membership_seed=spec.partition_seed,
            training_seed=spec.training_seed,
            train_records=train_records,
            validation_records=validation_records,
            eval_positions=tuple(int(item) for item in split["ordered_eval_positions"]),
            validation_order_provenance_id=str(split["validation_order_provenance_id"]),
        )
        if not isinstance(setup, dict):
            raise FollowupV1r2Error("SMOKE_SETUP_RESULT_TYPE_MISMATCH")
        _validate_setup_result(setup)
        loss, outputs = _prepare_smoke_loss(classifier)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if tuple(logits.shape[-1:]) != (2,) or not bool(classifier.torch.isfinite(logits).all().item()):
            raise FollowupV1r2Error("NUMERIC_SMOKE_LOGIT_CONTRACT_MISMATCH")
        if not bool(classifier.torch.isfinite(loss).item()):
            raise FollowupV1r2Error("NUMERIC_SMOKE_LOSS_NONFINITE")
        loss.backward()
        gradients = _gradient_summary(classifier)
        if classifier.trainer.state.global_step != 0:
            raise FollowupV1r2Error("NUMERIC_SMOKE_OPTIMIZER_STEP_MISMATCH")
        report = {
            "schema_version": "phi3_pooling_fixed4_followup_numeric_smoke_v1r2r4",
            "status": "PASS",
            "mode": "GPU_NUMERIC_SMOKE",
            "numeric_smoke_status": "PASS",
            "condition_id": CONDITION_ID,
            "model_id": pooling.MODEL_ID,
            "model_revision": pooling.MODEL_REVISION,
            "fold_index": spec.fold_index,
            "partition_seed": spec.partition_seed,
            "training_seed": spec.training_seed,
            "model_loaded": True,
            "trainer_created": True,
            "full_training_started": False,
            "forward_started": True,
            "forward_call_count": 1,
            "backward_started": True,
            "backward_call_count": 1,
            "prediction_started": False,
            "checkpoint_written": False,
            "actual_optimizer_steps": 0,
            "scheduler_steps": 0,
            "label_smoothing_factor": 0.05,
            "loss_finite": True,
            "loss_value": float(loss.detach().item()),
            "logits_finite": True,
            "logits_last_dimension": 2,
            "gradient_finite": True,
            "gradient_groups": gradients,
            "fixed_layer_weights": [0.25, 0.25, 0.25, 0.25],
            "fixed_layer_weights_trainable": False,
            "total_parameter_count": setup["total_parameter_count"],
            "trainable_parameter_count": setup["trainable_parameter_count"],
            "trainable_parameter_groups": setup["trainable_parameter_groups"],
            "runtime_versions": dict(EXPECTED_RUNTIME_VERSIONS),
            "peak_vram_mib": float(classifier.torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "elapsed_seconds": float(time.perf_counter() - started),
            **links,
            "official_test_accessed": False,
            "reserve_1_accessed": False,
            "reserve_2_accessed": False,
            "test_dataset_access_count": 0,
            "automatic_retry_attempted": False,
        }
        create_json_exclusive(target / "numeric_smoke_report.json", report)
        return report
    except BaseException as error:
        code, diagnostic_path, diagnostic_sha256 = _diagnostic_failure(
            error=error, recorder=recorder, root=target, links=links, classifier=classifier
        )
        if not (target / "numeric_smoke_report.json").exists():
            create_json_exclusive(
                target / "numeric_smoke_report.json",
                {
                    "schema_version": "phi3_pooling_fixed4_followup_numeric_smoke_v1r2r4",
                    "status": "INCOMPLETE_FAIL_CLOSED",
                    "mode": "GPU_NUMERIC_SMOKE",
                    "numeric_smoke_status": "INCOMPLETE_FAIL_CLOSED",
                    "error_code": code,
                    "diagnostic_exception_path": diagnostic_path,
                    "diagnostic_exception_sha256": diagnostic_sha256,
                    "actual_optimizer_steps": 0,
                    "scheduler_steps": 0,
                    "prediction_started": False,
                    "checkpoint_written": False,
                    **links,
                    "official_test_accessed": False,
                    "reserve_1_accessed": False,
                    "reserve_2_accessed": False,
                    "test_dataset_access_count": 0,
                    "automatic_retry_attempted": False,
                },
            )
        raise FollowupV1r2Error(code) from None
    finally:
        classifier.cleanup()
        gc.collect()


def _worker_spec(run_index: int) -> Any:
    match = next((item for item in v1.build_worker_specs() if item.run_index == run_index), None)
    if match is None:
        raise FollowupV1r2Error("WORKER_SPEC_NOT_IN_LOCKED_MATRIX")
    return match


def _expected_worker_root(paths: RuntimePaths, spec: Any) -> Path:
    return paths.runtime_root / "workers" / (
        f"run_{spec.run_index:03d}_{CONDITION_ID}_repeat_{spec.repetition}_"
        f"direction_{spec.direction}_replicate_{spec.replicate}_seed_{spec.training_seed}"
    )


def _worker_payload(result: Any, spec: Any) -> dict[str, Any]:
    payload = pooling.worker_payload(result)
    payload.update(
        {
            "schema_version": "phi3_pooling_fixed4_followup_worker_result_v1r2r4",
            "run_index": spec.run_index,
            "condition_id": CONDITION_ID,
            "display_label": DISPLAY_LABEL,
            "evidence_role": "prospectively_locked_follow_up_ablation",
            "repetition": spec.repetition,
            "direction": spec.direction,
            "replicate": spec.replicate,
            "partition_seed": spec.partition_seed,
            "temporal_execution_separate_from_prior_90": True,
        }
    )
    return payload


def _worker_split(paths: RuntimePaths, spec: Any) -> tuple[Mapping[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    plan = load_json(paths.project_root / v1.OLD_FOLD_PLAN_RELATIVE, "OLD_FOLD_PLAN_INVALID")
    records = prior.load_development_records(
        prior.RuntimePaths(paths.project_root, paths.hpo_artifact_root, paths.runtime_root)
    )
    prior.validate_fold_plan(plan, records)
    split = prior._plan_split(plan, spec.repetition, spec.direction)
    train_records = tuple(records[int(item)] for item in split["train_positions"])
    validation_records = tuple(records[int(item)] for item in split["ordered_eval_positions"])
    return split, train_records, validation_records


def validate_worker_payload(payload: Mapping[str, Any], spec: Any, split: Mapping[str, Any]) -> None:
    required = {
        "schema_version": "phi3_pooling_fixed4_followup_worker_result_v1r2r4",
        "run_index": spec.run_index,
        "condition_id": CONDITION_ID,
        "display_label": DISPLAY_LABEL,
        "repetition": spec.repetition,
        "direction": spec.direction,
        "replicate": spec.replicate,
        "partition_seed": spec.partition_seed,
        "training_seed": spec.training_seed,
        "validation_record_count": 750,
        "validation_order_provenance_id": split["validation_order_provenance_id"],
        "label_sequence_sha256": split["eval_label_sequence_sha256"],
        "actual_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "prediction_rule": "numpy.argmax(axis=-1)",
        "threshold_tuning": False,
        "final_test_accessed": False,
        "reserve_1_accessed": False,
        "reserve_2_accessed": False,
        "test_dataset_access_count": 0,
        "total_parameter_count": EXPECTED_PARAMETER_CONTRACT["total_parameter_count"],
        "trainable_parameter_count": EXPECTED_PARAMETER_CONTRACT["trainable_parameter_count"],
        "trainable_parameter_groups": EXPECTED_PARAMETER_CONTRACT["trainable_parameter_groups"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise FollowupV1r2Error("WORKER_RESULT_CONTRACT_MISMATCH")
    if payload.get("ordered_eval_positions") != split["ordered_eval_positions"]:
        raise FollowupV1r2Error("WORKER_EVAL_ORDER_MISMATCH")
    if any(len(payload.get(key, [])) != 750 for key in ("labels", "probabilities", "predictions")):
        raise FollowupV1r2Error("WORKER_OUTPUT_LENGTH_MISMATCH")
    if pooling.recursively_has_forbidden_content(payload):
        raise FollowupV1r2Error("WORKER_FORBIDDEN_CONTENT")


def run_worker(paths: RuntimePaths, worker_root: Path, run_index: int, expected_hashes: Mapping[str, str]) -> int:
    _, dry_hash = verify_cpu_artifacts(paths)
    _verify_setup_report(paths, dry_hash)
    _verify_smoke_report(paths, dry_hash)
    observed = _expected_report_links(paths, dry_hash)
    if observed != dict(expected_hashes):
        raise FollowupV1r2Error("WORKER_EXPECTED_HASH_MISMATCH")
    spec = _worker_spec(run_index)
    if not worker_root.is_dir() or not _same_path(worker_root, _expected_worker_root(paths, spec)):
        raise FollowupV1r2Error("WORKER_ROOT_CONTRACT_MISMATCH")
    recorder = pooling.DiagnosticStageRecorder(
        worker_root, condition_id=CONDITION_ID, fold_index=spec.fold_index, training_seed=spec.training_seed
    )
    classifier = LockedP2Classifier(paths.project_root)
    try:
        split, train_records, validation_records = _worker_split(paths, spec)
        result = classifier.evaluate_pooling_diagnostic(
            recorder=recorder,
            setup_probe=False,
            condition=v1.p2_condition_spec(),
            fold_index=spec.fold_index,
            fold_membership_seed=spec.partition_seed,
            training_seed=spec.training_seed,
            train_records=train_records,
            validation_records=validation_records,
            eval_positions=tuple(int(item) for item in split["ordered_eval_positions"]),
            validation_order_provenance_id=str(split["validation_order_provenance_id"]),
        )
        payload = _worker_payload(result, spec)
        validate_worker_payload(payload, spec, split)
        result_path = worker_root / "worker_result.json"
        create_json_exclusive(result_path, payload)
        recorder.record("worker_result_persisted", torch_module=classifier.torch)
        create_json_exclusive(
            worker_root / "worker_status.json",
            {
                "status": "PASS",
                "run_index": spec.run_index,
                "condition_id": CONDITION_ID,
                "repetition": spec.repetition,
                "direction": spec.direction,
                "replicate": spec.replicate,
                "training_seed": spec.training_seed,
                "actual_optimizer_steps": result.actual_optimizer_steps,
                "worker_result_sha256": sha256_file(result_path),
                "official_test_accessed": False,
                "reserve_1_accessed": False,
                "reserve_2_accessed": False,
                "test_dataset_access_count": 0,
                "automatic_retry_attempted": False,
            },
        )
        return 0
    except BaseException as error:
        code, diagnostic_path, diagnostic_sha256 = _diagnostic_failure(
            error=error, recorder=recorder, root=worker_root, links=observed, classifier=classifier
        )
        if not (worker_root / "worker_status.json").exists():
            create_json_exclusive(
                worker_root / "worker_status.json",
                {
                    "status": "INCOMPLETE_FAIL_CLOSED",
                    "error_code": code,
                    "run_index": spec.run_index,
                    "condition_id": CONDITION_ID,
                    "last_successful_stage": recorder.last_successful_stage,
                    "failed_stage": recorder.current_stage,
                    "diagnostic_exception_path": diagnostic_path,
                    "diagnostic_exception_sha256": diagnostic_sha256,
                    "official_test_accessed": False,
                    "reserve_1_accessed": False,
                    "reserve_2_accessed": False,
                    "test_dataset_access_count": 0,
                    "automatic_retry_attempted": False,
                },
            )
        return 1
    finally:
        classifier.cleanup()
        gc.collect()


def _old_payloads(project_root: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for artifact in v1.collect_old_worker_artifacts(project_root):
        payload = load_json(
            project_root / artifact["worker_result_relative_path"], "OLD_WORKER_RESULT_INVALID"
        )
        key = (
            str(payload["condition_id"]),
            int(payload["repetition"]),
            int(payload["direction"]),
            int(payload["replicate"]),
        )
        rows[key] = payload
    if len(rows) != 90:
        raise FollowupV1r2Error("OLD_WORKER_PAYLOAD_COUNT_MISMATCH")
    return rows


def _aligned(payloads: Sequence[Mapping[str, Any]]) -> bool:
    return bool(payloads) and all(
        item.get("validation_order_provenance_id") == payloads[0].get("validation_order_provenance_id")
        and item.get("ordered_eval_positions") == payloads[0].get("ordered_eval_positions")
        and item.get("labels") == payloads[0].get("labels")
        for item in payloads[1:]
    )


def _condition_scores(payloads: Mapping[tuple[str, int, int, int], Mapping[str, Any]], condition: str) -> dict[str, Any]:
    split_means = {
        (repetition, direction): statistics.fmean(
            float(payloads[(condition, repetition, direction, replicate)]["metrics"]["macro_f1"])
            for replicate in range(1, 4)
        )
        for repetition in range(1, 6)
        for direction in (1, 2)
    }
    repetition_means = {
        repetition: statistics.fmean((split_means[(repetition, 1)], split_means[(repetition, 2)]))
        for repetition in range(1, 6)
    }
    return {
        "condition_id": condition,
        "split_mean_macro_f1": {
            f"repeat_{r}_direction_{d}": split_means[(r, d)]
            for r in range(1, 6)
            for d in (1, 2)
        },
        "repetition_mean_macro_f1": [repetition_means[r] for r in range(1, 6)],
        "overall_mean_macro_f1": statistics.fmean(repetition_means.values()),
    }


def _paired_contrast(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> dict[str, Any]:
    deltas = [
        float(l) - float(r)
        for l, r in zip(left["repetition_mean_macro_f1"], right["repetition_mean_macro_f1"], strict=True)
    ]
    mean_delta = statistics.fmean(deltas)
    standard_deviation = statistics.stdev(deltas)
    margin = T_CRITICAL_95_DF4 * standard_deviation / math.sqrt(5.0)
    return {
        "contrast": label,
        "analysis_role": "descriptive_follow_up",
        "unit": "repetition",
        "n": 5,
        "repetition_deltas": deltas,
        "mean_repetition_delta": mean_delta,
        "sample_standard_deviation": standard_deviation,
        "student_t_95_ci": {"lower": mean_delta - margin, "upper": mean_delta + margin},
        "degrees_of_freedom": 4,
        "binary_significance_gate_applied": False,
        "selection_or_promotion_applied": False,
    }


def _write_incomplete(
    paths: RuntimePaths,
    error_code: str,
    completed: Sequence[Mapping[str, Any]],
    failed_spec: Any | None,
    started: float,
) -> None:
    path = paths.runtime_root / "p2_eq4_followup_incomplete_report.json"
    if path.exists():
        return
    create_json_exclusive(
        path,
        {
            "schema_version": "phi3_pooling_fixed4_followup_incomplete_v1r2r4",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error_code": error_code,
            "failed_worker": asdict(failed_spec) if failed_spec is not None else None,
            "completed_worker_count": len(completed),
            "completed_workers": list(completed),
            "total_elapsed_seconds": float(time.perf_counter() - started),
            "automatic_retry_attempted": False,
            "official_test_accessed": False,
            "reserve_1_accessed": False,
            "reserve_2_accessed": False,
            "test_dataset_access_count": 0,
        },
    )


def run_real_30(paths: RuntimePaths) -> dict[str, Any]:
    _, dry_hash = verify_cpu_artifacts(paths)
    _verify_setup_report(paths, dry_hash)
    _verify_smoke_report(paths, dry_hash)
    workers_root = paths.runtime_root / "workers"
    final_path = paths.runtime_root / "p2_eq4_followup_report.json"
    incomplete_path = paths.runtime_root / "p2_eq4_followup_incomplete_report.json"
    if workers_root.exists() or final_path.exists() or incomplete_path.exists():
        raise FollowupV1r2Error("REAL_30_RUNTIME_ARTIFACT_ALREADY_EXISTS")
    workers_root.mkdir(exist_ok=False)
    links = _expected_report_links(paths, dry_hash)
    completed: list[dict[str, Any]] = []
    new_payloads: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    started = time.perf_counter()
    failed_spec: Any | None = None
    try:
        for spec in v1.build_worker_specs():
            failed_spec = spec
            worker_root = _expected_worker_root(paths, spec)
            worker_root.mkdir(exist_ok=False)
            command = [
                str(FIXED_PYTHON),
                "-B",
                str(Path(__file__).resolve()),
                "--project-root", str(paths.project_root),
                "--hpo-artifact-root", str(paths.hpo_artifact_root),
                "--runtime-root", str(paths.runtime_root),
                "--confirm-real-30-worker-p2-eq4-v1r2r4",
                "--worker",
                "--worker-root", str(worker_root),
                "--run-index", str(spec.run_index),
                "--expected-runtime-sha256", links["runtime_source_sha256"],
                "--expected-verifier-sha256", links["cpu_verifier_sha256"],
                "--expected-dry-gate-sha256", links["dry_gate_report_sha256"],
            ]
            process = subprocess.run(command, check=False)
            if process.returncode != 0:
                status_path = worker_root / "worker_status.json"
                status = load_json(status_path, "WORKER_STATUS_INVALID") if status_path.is_file() else {}
                raise FollowupV1r2Error(str(status.get("error_code", "WORKER_PROCESS_FAILED")))
            result_path = worker_root / "worker_result.json"
            payload = load_json(result_path, "WORKER_RESULT_INVALID")
            split, _, _ = _worker_split(paths, spec)
            validate_worker_payload(payload, spec, split)
            key = (CONDITION_ID, spec.repetition, spec.direction, spec.replicate)
            new_payloads[key] = payload
            completed.append(
                {
                    "run_index": spec.run_index,
                    "condition_id": CONDITION_ID,
                    "repetition": spec.repetition,
                    "direction": spec.direction,
                    "replicate": spec.replicate,
                    "training_seed": spec.training_seed,
                    "macro_f1": float(payload["metrics"]["macro_f1"]),
                    "worker_result_sha256": sha256_file(result_path),
                }
            )
            failed_spec = None
        # Detect any mutation during the long sequential run before using prior controls.
        verify_v1r1_preserved(paths.project_root)
        old = _old_payloads(paths.project_root)
        combined = {**old, **new_payloads}
        for repetition in range(1, 6):
            for direction in (1, 2):
                for replicate in range(1, 4):
                    aligned = [
                        combined[(condition, repetition, direction, replicate)]
                        for condition in ("P0", "P1", "P5", CONDITION_ID)
                    ]
                    if not _aligned(aligned):
                        raise FollowupV1r2Error("ORDER_LABEL_POSITION_ALIGNMENT_MISMATCH")
        summaries = {
            condition: _condition_scores(combined, condition)
            for condition in ("P0", "P1", CONDITION_ID, "P5")
        }
        contrasts = {
            "P5_minus_P2_EQ4": _paired_contrast(summaries["P5"], summaries[CONDITION_ID], "P5_minus_P2_EQ4"),
            "P2_EQ4_minus_P1": _paired_contrast(summaries[CONDITION_ID], summaries["P1"], "P2_EQ4_minus_P1"),
            "P1_minus_P0": _paired_contrast(summaries["P1"], summaries["P0"], "P1_minus_P0"),
            "P5_minus_P0": _paired_contrast(summaries["P5"], summaries["P0"], "P5_minus_P0"),
        }
        report = {
            "schema_version": "phi3_pooling_fixed4_followup_report_v1r2r4",
            "status": "PASS",
            "runtime_version": RUNTIME_VERSION,
            "evidence_role": "prospectively_locked_follow_up_ablation",
            "condition_id": CONDITION_ID,
            "display_label": DISPLAY_LABEL,
            "worker_status": "30_OF_30_PASS",
            "new_worker_count": 30,
            "frozen_prior_worker_count": 90,
            "combined_worker_count": 120,
            "condition_summaries": summaries,
            "contrasts": contrasts,
            "primary_follow_up_contrast": "P5_minus_P2_EQ4",
            "secondary_follow_up_contrast": "P2_EQ4_minus_P1",
            "analysis_unit": "five_repetitions",
            "binary_significance_gate_applied": False,
            "selection_or_promotion_applied": False,
            "temporal_cuda_nondeterminism_residual_limit": True,
            "old_and_new_workers_contemporaneous": False,
            "worker_results": completed,
            "parameter_contract": EXPECTED_PARAMETER_CONTRACT,
            "prediction_rule": "numpy.argmax(axis=-1)",
            "threshold_tuning": False,
            "total_elapsed_seconds": float(time.perf_counter() - started),
            "peak_vram_mib": max(float(item["peak_vram_mib"]) for item in new_payloads.values()),
            **links,
            "official_test_accessed": False,
            "reserve_1_accessed": False,
            "reserve_2_accessed": False,
            "test_dataset_access_count": 0,
            "automatic_retry_attempted": False,
        }
        create_json_exclusive(final_path, report)
        return report
    except BaseException as error:
        code = error.code if isinstance(error, FollowupV1r2Error) else type(error).__name__
        _write_incomplete(paths, code, completed, failed_spec, started)
        raise FollowupV1r2Error(code) from None


def _paths(arguments: argparse.Namespace) -> RuntimePaths:
    project_root = arguments.project_root.resolve()
    locked = locked_paths(project_root)
    hpo = arguments.hpo_artifact_root.resolve() if arguments.hpo_artifact_root else locked.hpo_artifact_root
    runtime = arguments.runtime_root.resolve() if arguments.runtime_root else locked.runtime_root
    paths = RuntimePaths(project_root, hpo, runtime)
    verify_runtime_paths(paths)
    return paths


def blocked_status() -> None:
    print("P2_EQ4_FOLLOWUP_V1R2R4_STATUS=BLOCKED")
    print("REASON=EXPLICIT_GPU_SETUP_PROBE_AUTHORIZATION_REQUIRED")
    print("GPU_SETUP_PROBE_STATUS=NOT_STARTED")
    print("GPU_NUMERIC_SMOKE_STATUS=NOT_STARTED")
    print("REAL_30_WORKER_STATUS=NOT_STARTED")
    print("MODEL_LOAD_STARTED=false")
    print("TRAINING_STARTED=false")
    print("FORWARD_STARTED=false")
    print("BACKWARD_STARTED=false")
    print("PREDICTION_STARTED=false")
    print("ACTUAL_OPTIMIZER_STEPS=0")
    print("SCHEDULER_STEPS=0")
    print("PROTECTED_TEST_ACCESS_STATUS=NOT_ACCESSED")
    print("NO_EXISTING_ARTIFACT_OVERWRITTEN=true")
    print("NO_GIT_WRITE=true")
    print("NO_AUTOMATIC_RETRY=true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--hpo-artifact-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--prepare-cpu-only", action="store_true")
    parser.add_argument("--confirm-gpu-setup-probe-v1r2r4", action="store_true")
    parser.add_argument("--confirm-gpu-numeric-smoke-v1r2r4", action="store_true")
    parser.add_argument("--confirm-real-30-worker-p2-eq4-v1r2r4", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--expected-runtime-sha256")
    parser.add_argument("--expected-verifier-sha256")
    parser.add_argument("--expected-dry-gate-sha256")
    arguments = parser.parse_args(argv)
    modes = (
        arguments.prepare_cpu_only,
        arguments.confirm_gpu_setup_probe_v1r2r4,
        arguments.confirm_gpu_numeric_smoke_v1r2r4,
        arguments.confirm_real_30_worker_p2_eq4_v1r2r4,
    )
    if sum(bool(item) for item in modes) != 1:
        if not any(modes) and not arguments.worker:
            blocked_status()
            return 3
        return 2
    paths = _paths(arguments)
    if arguments.prepare_cpu_only:
        if arguments.worker:
            return 2
        result = prepare_cpu_only(paths.project_root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if arguments.worker:
        if not arguments.confirm_real_30_worker_p2_eq4_v1r2r4 or arguments.worker_root is None or arguments.run_index is None:
            return 2
        expected = {
            "runtime_source_sha256": arguments.expected_runtime_sha256,
            "cpu_verifier_sha256": arguments.expected_verifier_sha256,
            "dry_gate_report_sha256": arguments.expected_dry_gate_sha256,
        }
        if any(value is None for value in expected.values()):
            return 2
        return run_worker(paths, arguments.worker_root.resolve(), arguments.run_index, expected)
    if arguments.confirm_gpu_setup_probe_v1r2r4:
        report = run_gpu_setup_probe(paths)
        print("P2_EQ4_GPU_SETUP_PROBE_V1R2R4_STATUS=PASS")
        print(f"STOP_STAGE={report['stop_stage']}")
        print("ACTUAL_OPTIMIZER_STEPS=0")
        return 0
    if arguments.confirm_gpu_numeric_smoke_v1r2r4:
        report = run_gpu_numeric_smoke(paths)
        print("P2_EQ4_GPU_NUMERIC_SMOKE_V1R2R4_STATUS=PASS")
        print(f"FORWARD_CALL_COUNT={report['forward_call_count']}")
        print(f"BACKWARD_CALL_COUNT={report['backward_call_count']}")
        print("ACTUAL_OPTIMIZER_STEPS=0")
        return 0
    report = run_real_30(paths)
    print("P2_EQ4_REAL_30_WORKER_V1R2R4_STATUS=PASS")
    print(f"WORKER_STATUS={report['worker_status']}")
    print("PROTECTED_TEST_NOT_ACCESSED")
    return 0


def validate_static_contract() -> None:
    if v1.CONDITION_ID != CONDITION_ID or len(v1.build_worker_specs()) != EXPECTED_WORKERS:
        raise RuntimeError("P2_V1_SCIENTIFIC_REUSE_CONTRACT_INVALID")
    if pooling.MODEL_ID != prior.MODEL_ID or pooling.MODEL_REVISION != prior.MODEL_REVISION:
        raise RuntimeError("MODEL_IDENTITY_REUSE_CONTRACT_INVALID")
    if EXPECTED_OPTIMIZER_STEPS != 564:
        raise RuntimeError("WORKER_BUDGET_CONTRACT_INVALID")


validate_static_contract()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FollowupV1r2Error as error:
        print("P2_EQ4_FOLLOWUP_V1R2R4_STATUS=INCOMPLETE_FAIL_CLOSED")
        print(f"ERROR_CODE={error.code}")
        print("PROTECTED_TEST_NOT_ACCESSED")
        print("NO_AUTOMATIC_RETRY=true")
        raise SystemExit(1) from None
