#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_NAME="${ASR_BENCHMARK_BREV_WORKSPACE:-cm-nakamura-asr-benchmark}"
INSTANCE_TYPE="${ASR_BENCHMARK_BREV_TYPE:-l40s-48gb.1x}"

WORKSPACE_STATUS="$(
  brev list --json | jq -r --arg name "${WORKSPACE_NAME}" \
    '.workspaces[] | select(.name == $name) | .status' | head -1
)"

if [[ -n "${WORKSPACE_STATUS}" ]]; then
  echo "workspace already exists: ${WORKSPACE_NAME} (${WORKSPACE_STATUS})"
  if [[ "${WORKSPACE_STATUS}" == "STOPPED" ]]; then
    brev start "${WORKSPACE_NAME}"
  fi
else
  brev create "${WORKSPACE_NAME}" --type "${INSTANCE_TYPE}"
fi

for _ in $(seq 1 40); do
  WORKSPACE_JSON="$(brev list --json)"
  STATUS="$(jq -r --arg name "${WORKSPACE_NAME}" '.workspaces[] | select(.name == $name) | .status' <<<"${WORKSPACE_JSON}")"
  BUILD_STATUS="$(jq -r --arg name "${WORKSPACE_NAME}" '.workspaces[] | select(.name == $name) | .build_status' <<<"${WORKSPACE_JSON}")"
  SHELL_STATUS="$(jq -r --arg name "${WORKSPACE_NAME}" '.workspaces[] | select(.name == $name) | .shell_status' <<<"${WORKSPACE_JSON}")"
  if [[ "${STATUS}" == "RUNNING" && "${BUILD_STATUS}" == "COMPLETED" && "${SHELL_STATUS}" == "READY" ]]; then
    echo "workspace ready: ${WORKSPACE_NAME}"
    exit 0
  fi
  if [[ "${STATUS}" == "FAILURE" ]]; then
    echo "workspace creation failed: ${WORKSPACE_NAME}" >&2
    exit 1
  fi
  sleep 30
done

echo "workspace did not become ready within 20 minutes: ${WORKSPACE_NAME}" >&2
exit 1
