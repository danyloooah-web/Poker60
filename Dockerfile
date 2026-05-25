FROM ghcr.io/subnet112/solver-base:v1

# solver-base already ships web3 and core deps; pip may fail offline during screening.
COPY requirements.txt /app/solver/requirements.txt
RUN pip install --no-cache-dir -r /app/solver/requirements.txt 2>/dev/null || true

COPY solver.py /app/solver/solver.py
COPY strategies /app/solver/strategies
COPY common /app/solver/common
COPY draft /app/solver/draft

# Do NOT add CMD or ENTRYPOINT — the base image handles that.
