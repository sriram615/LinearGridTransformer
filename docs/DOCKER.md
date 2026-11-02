# Local Docker setup

This Compose deployment runs the FastAPI inference API and the production React frontend. The backend reads a separate, read-only Tetouan runtime artifact bundle. MLflow is optional tracking infrastructure and is not part of the default Compose stack or a backend dependency.

## Prerequisites

- Docker Engine/Desktop with Docker Compose v2.
- About 20 MB of local artifact storage, plus image space.
- Node/pnpm are not needed on the host; the frontend image builds with Node 22.12.0 and pnpm 10.34.3.

## Runtime artifact bundle

The bundle mirrors the required subset of the repository:

    runtime-artifacts/
      data/processed/tetouan/{metadata.json,series.npz}
      checkpoints/tetouan/{two original checkpoints}
      checkpoints/tetouan/extended_training_v2/{six seed/model checkpoints}
      results/tetouan/tetouan_t4_20261101_v1/{manifest,history,evaluation JSON,visualisations}
      results/tetouan/extended_training_v2/{three manifests,three histories,evaluation_test_v1 contents}

The checked-in Docker build context excludes the source data, full result/checkpoint trees and processed household data. The separate runtime-artifacts directory remains available for the read-only bind mount. No raw CSV/ZIP, household checkpoint/data, local MLflow state or training-only smoke checkpoint belongs in this bundle.

To prepare a fresh bundle from the repository, first choose a new destination that does not already exist, then copy only the required files. For example:

    BUNDLE=runtime-artifacts-next
    test ! -e "$BUNDLE"
    mkdir -p "$BUNDLE/data/processed/tetouan" "$BUNDLE/checkpoints/tetouan" "$BUNDLE/results/tetouan"
    cp -p data/processed/tetouan/metadata.json data/processed/tetouan/series.npz "$BUNDLE/data/processed/tetouan/"
    cp -p checkpoints/tetouan/tetouan_t4_20261101_v1_{lineargridtransformer,cdlinear}.pt "$BUNDLE/checkpoints/tetouan/"
    cp -R checkpoints/tetouan/extended_training_v2 "$BUNDLE/checkpoints/tetouan/"
    cp -R results/tetouan/tetouan_t4_20261101_v1 results/tetouan/extended_training_v2 "$BUNDLE/results/tetouan/"

After verifying the new bundle, set RUNTIME_ARTIFACTS_DIR to it in .env. Do not copy into an existing bundle; use a new versioned destination for updates.

## Configure, build and start

    cp .env.example .env
    docker compose config
    docker compose build
    docker compose up -d

The frontend is available at http://localhost:8080/. The proxied readiness URL is http://localhost:8080/health. The frontend port binds only to loopback, and the backend is internal to the Compose network with no host port published.

The API receives artifact paths under /opt/runtime-artifacts from Compose environment variables. The bundle mount is read-only and can be replaced by selecting a new versioned bundle path.

## Logs, shutdown and rebuild

    docker compose logs -f backend frontend
    docker compose down
    docker compose build --no-cache
    docker compose up -d

The API is configured for CPU inference and one Uvicorn worker. Nginx serves the Vite production build and forwards /api/ and /health to backend:8000, so browser requests are same-origin and require no CORS allowance.

## MLflow

The existing MLflow database and artifact store remain separate and are not mounted into the application containers. FastAPI loads model checkpoints and metadata from the runtime artifact bundle and must work when MLflow is unavailable. The local SQLite store contains absolute artifact URIs; moving it into a container requires a separate migration/path plan.
