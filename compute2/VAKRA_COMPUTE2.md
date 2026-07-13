# VAKRA on Compute2

This path avoids requiring `docker exec` from inside the job. The runner starts
VAKRA's MCP server as a local subprocess inside the same container with:

```bash
--mcp-launcher local --mcp-command python --mcp-args /app/mcp_dispatch.py
```

## 1. Probe Compute2

Submit the probe first. It checks modules, container runtimes, Python, CUDA, and
a small optional container smoke from inside an allocated job:

```bash
sbatch compute2/compute2_container_probe.sbatch
```

## 2. Build and Push an AMD64 Image

From a machine with Docker buildx:

```bash
IMAGE=ghcr.io/<user>/vakra-ensemble:amd64 \
BASE_IMAGE=ghcr.io/<user>/vakra-base:amd64 \
PUSH=1 \
compute2/build_vakra_ensemble_image.sh
```

The base image is VAKRA's unified MCP image. The overlay image adds:

- `compute2/vakra_mcp_ensemble_runner.py`
- `compute2/tau_ensemble_agent.py`
- model runtime packages needed by the GPT-OSS adapter bank

## 3. Run a Smoke

After `external/vakra` and the VAKRA data are present on Storage3:

```bash
sbatch --export=ALL,VAKRA_IMAGE=ghcr.io/<user>/vakra-ensemble:amd64,DOMAIN=beer_factory,CAPABILITY_ID=1,MAX_SAMPLES=1 \
  compute2/compute2_vakra_ensemble.sbatch
```

Use `RUNNER_KIND=apptainer`, `RUNNER_KIND=singularity`, or `RUNNER_KIND=docker`
if the probe shows a specific runtime. Leave it as `auto` otherwise.

## Notes

- `MAX_SAMPLES=0` means run the full domain file.
- Capability 1 does not need background VAKRA services.
- Capabilities 2, 3, and 4 start `/app/entrypoint.sh` in the background before
  launching the MCP subprocess.
- Override `VAKRA_ROOT`, `VAKRA_DB_ROOT`, `VAKRA_CHROMA_ROOT`, and
  `VAKRA_QUERIES_ROOT` if the Storage3 layout differs.
