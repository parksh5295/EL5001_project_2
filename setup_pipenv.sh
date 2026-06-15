#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PIPENV_VENV_IN_PROJECT=1

if python3 -m pipenv --version >/dev/null 2>&1; then
  PIPENV=(python3 -m pipenv)
elif command -v pipenv >/dev/null 2>&1; then
  PIPENV=(pipenv)
else
  echo "pipenv를 찾을 수 없습니다." >&2
  echo "먼저 실행하세요: python3 -m pip install --user pipenv" >&2
  echo "그리고 PATH 추가: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

if [[ -n "${PIPENV_PYTHON:-}" ]]; then
  PYTHON_BIN="$PIPENV_PYTHON"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.10)"
elif command -v pyenv >/dev/null 2>&1; then
  PY310_VER="$(pyenv versions --bare | awk '/^3\.10(\.|$)/{v=$0} END{print v}')"
  if [[ -n "$PY310_VER" ]]; then
    PYTHON_BIN="$(PYENV_VERSION="$PY310_VER" pyenv which python)"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "사용 가능한 python3 실행 파일을 찾지 못했습니다." >&2
  echo "환경변수로 지정해 주세요: PIPENV_PYTHON=/path/to/python ./setup_pipenv.sh" >&2
  exit 1
fi

IS_PY310="false"
if "$PYTHON_BIN" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)" >/dev/null 2>&1; then
  IS_PY310="true"
else
  echo "[warn] Python 3.10을 찾지 못해 fallback 사용: $PYTHON_BIN"
  echo "[warn] Pipfile 요구 버전(3.10)과 다를 수 있지만, sudo 없는 환경에서 진행합니다."
fi

if [[ -d ".venv" ]]; then
  echo "[prep] 기존 .venv 삭제 후 재생성"
  rm -rf ".venv"
fi

echo "[1/2] pipenv 가상환경 생성 (${PYTHON_BIN})"
"${PIPENV[@]}" --clear >/dev/null
"${PIPENV[@]}" --python "$PYTHON_BIN" >/dev/null

if [[ "$IS_PY310" == "true" ]]; then
  echo "[2/2] Pipfile 의존성 설치"
  "${PIPENV[@]}" install
else
  FALLBACK_REQ="${ROOT_DIR}/requirements-fallback.txt"
  if [[ ! -f "$FALLBACK_REQ" ]]; then
    echo "fallback requirements 파일이 없습니다: $FALLBACK_REQ" >&2
    exit 1
  fi
  echo "[2/2] fallback 의존성 설치 (${FALLBACK_REQ})"
  "${PIPENV[@]}" run python -m pip install --upgrade pip setuptools wheel
  "${PIPENV[@]}" run python -m pip install -r "$FALLBACK_REQ"
fi

echo "완료: ${ROOT_DIR}/.venv"
"${PIPENV[@]}" run python -c "import sys, numpy, six, hexdump; print('Venv Python:', sys.executable); print('numpy:', numpy.__version__)"
