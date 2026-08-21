# Runtime Pack Cold-Start Build

This document is the pack handoff contract for agentic Slime. It covers a
clean build when no previous runtime pack, data pack, conda prefix, or
cluster-specific NAS mount is available.

## Guarantees And Limits

- `ROOT_DIR` may be any writable absolute path on local disk, a shared
  filesystem, or a mounted cloud filesystem. The build does not require the
  historical `/mnt/bn/...` trees.
- Runtime packs are built from the checked-out agentic-slime repository,
  pinned public source commits, public package indexes, and public dataset
  downloaders.
- A source rebuild is intended to reproduce a compatible runtime. It is not a
  promise of byte-identical output forever because public conda and pip indexes
  can change or remove transitive artifacts.
- After verification, back up the complete pack bundle to any durable shared
  or cloud storage available on the target organization. The backup is the
  preferred exact restore path; source rebuilding is the recovery path.
- Do not put API keys, SSH keys, W&B credentials, model weights, checkpoints,
  or private teacher traces inside public runtime packs.

## Layers

Keep these layers independent:

1. Source checkout: `${ROOT_DIR}/code/slime` and pinned upstream checkouts.
2. Build prefixes and caches: disposable; they can be deleted after publish.
3. Runtime packs: Slime, one task env per environment, and optional W&B.
4. Data packs: task datasets only; never hide data inside a runtime pack.
5. Models, secrets, runs, and checkpoints: separate durable assets.

The canonical runtime bundle for `NAME` is:

```text
${ROOT_DIR}/packs/NAME.tar.gz
${ROOT_DIR}/packs/NAME.tar.gz.sha256
${ROOT_DIR}/packs/NAME.revision
${ROOT_DIR}/packs/NAME.manifest.txt
```

The canonical data bundle uses the same four-member contract with
`NAME-data` as the basename.

Do not copy only the tarball. The checksum validates transfer integrity, the
revision identifies the artifact, and the manifest records source/package
provenance.

## Build Host Requirements

Use a dedicated build worker with adequate local disk and network access. Do
not use a small development jump host for environment-dependent work.

Required for all packs:

- Linux x86_64 or aarch64;
- `bash`, `git`, `curl`, `tar`, `bzip2`, and SHA-256 tools;
- access to GitHub, PyPI, conda-forge, and the public dataset endpoints;
- at least 50 GiB free for task envs and substantially more for task data.

The Slime foundation pack additionally requires:

- x86_64 Linux with an NVIDIA driver compatible with CUDA 12.9;
- a compiler/native build toolchain;
- roughly 150-200 GiB of temporary disk headroom;
- enough RAM and time to compile FlashAttention and other CUDA extensions.

Build on a machine compatible with the target GPU nodes. Do not build the
Slime CUDA pack on macOS or an unrelated CPU-only image and expect it to run on
the cluster.

## Bootstrap From An Empty Root

Choose any writable root. No historical pack directory is required.

```bash
export ROOT_DIR=/absolute/path/to/agentic-slime-workspace
mkdir -p "${ROOT_DIR}"/{code,data,envs,models,packs,runs,secrets,scripts,tools}

git clone https://github.com/Yilgrimage/agentic_slime.git \
  "${ROOT_DIR}/code/slime"
git -C "${ROOT_DIR}/code/slime" fetch origin agentic-env-backend
git -C "${ROOT_DIR}/code/slime" checkout --detach origin/agentic-env-backend
git -C "${ROOT_DIR}/code/slime" rev-parse HEAD

cp -a "${ROOT_DIR}/code/slime/.claude/skills/server-ops-discipline/scripts/." \
  "${ROOT_DIR}/scripts/"
chmod +x "${ROOT_DIR}/scripts/"*.sh
"${ROOT_DIR}/scripts/configure_root.sh" "${ROOT_DIR}" --yes

ROOT_DIR="${ROOT_DIR}" "${ROOT_DIR}/scripts/bootstrap_micromamba.sh"
```

Use a pushed, reviewed commit. Record `git rev-parse HEAD` in the handoff; do
not build from an uncommitted working tree. The unified builder refuses a dirty
agentic-slime checkout by default.

To install the same operational guidance into a new Codex home:

```bash
mkdir -p "${HOME}/.codex/skills"
rm -rf "${HOME}/.codex/skills/server-ops-discipline"
cp -a "${ROOT_DIR}/code/slime/.claude/skills/server-ops-discipline" \
  "${HOME}/.codex/skills/server-ops-discipline"
```

## Build Runtime Packs

Build lightweight task/W&B packs first:

```bash
ROOT_DIR="${ROOT_DIR}" \
  "${ROOT_DIR}/scripts/build_runtime_packs.sh" \
  --repo "${ROOT_DIR}/code/slime" \
  --runtime alfworld,appworld,tau2,webshop,wandb \
  --recreate
```

Pack builders ignore ambient user pip configuration and default to
`https://pypi.org/simple`. On a restricted network, set
`PACK_PIP_INDEX_URL` to an explicitly approved mirror; do not rely on an
operator's hidden `pip.conf`.

Build the Slime foundation pack separately because it is expensive:

```bash
ROOT_DIR="${ROOT_DIR}" \
  LOCAL_RUNTIME_DIR=/local/fast-disk/server-ops-runtime \
  "${ROOT_DIR}/scripts/build_runtime_packs.sh" \
  --repo "${ROOT_DIR}/code/slime" \
  --runtime slime \
  --recreate
```

`LOCAL_RUNTIME_DIR` should be node-local scratch. Do not route compiler caches,
conda package caches, or temporary archives to a slow NAS merely because the
final pack is durable there.

The Slime publisher always refreshes its base archive from the env built in the
same invocation. It does not silently reuse an older `${ROOT_DIR}/envs/archives`
artifact.

Current default source pins are:

| Component | Default pin |
| --- | --- |
| agentic Slime | the selected checkout commit |
| SGLang | `5a15cde858ea09b77116212a39356f2fc51b8584` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` |
| FlashQLA | `c18a4860ea9cb937f1075d606b4823d6ae34e880` |
| Megatron-Bridge | `923842f5d14ca9db2f243b2dfce01826176dd533` |
| ALFWorld | `0.4.2` |
| AppWorld | `0.1.3.post1` |
| tau2-bench | `1746a25db265724f6fed2260934bb67f1514fad7` |
| WebShop | `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` |
| W&B | `0.26.1` |

The manifest is authoritative for the artifact actually produced. If a pin is
intentionally overridden, preserve the generated manifest and record the
override in the experiment handoff.

## Verify Runtime Packs

First verify bundle integrity without extracting:

```bash
ROOT_DIR="${ROOT_DIR}" \
  "${ROOT_DIR}/scripts/verify_pack_bundle.sh" \
  --runtime slime,alfworld,appworld,tau2,webshop,wandb \
  --no-extract
```

Then run extraction, `conda-unpack`, and import smoke tests on a clean worker:

```bash
ROOT_DIR="${ROOT_DIR}" \
  TMPDIR=/local/fast-disk/pack-verify \
  "${ROOT_DIR}/scripts/verify_pack_bundle.sh" \
  --runtime slime,alfworld,appworld,tau2,webshop,wandb
```

A tarball that exists but lacks its checksum, revision, or manifest is not a
complete new-style pack bundle.

## Prepare Public Task Data

Runtime packs and data are deliberately separate. Build runtime packs before
running data downloaders.

ALFWorld and AppWorld use their official CLI download paths:

```bash
ROOT_DIR="${ROOT_DIR}" \
  LOCAL_ENVS_DIR="${ROOT_DIR}/envs" \
  "${ROOT_DIR}/scripts/prepare_data.sh" \
  --data alfworld,appworld \
  --validate-data-load
```

WebShop can be built without an existing data backup. The script downloads the
three official 100k files into a disposable staging checkout and builds only
the required `indexes_100k`; it does not dirty the pinned source checkout:

```bash
ROOT_DIR="${ROOT_DIR}" \
  WEBSHOP_DOWNLOAD=1 \
  WEBSHOP_PYTHON="${ROOT_DIR}/envs/webshop-clean/bin/python" \
  "${ROOT_DIR}/scripts/prepare_data.sh" \
  --data webshop
```

For tau2, the pinned upstream checkout supplies official domain data. The
repository's public AReaL downloader supplies the synthetic training data:

```bash
ROOT_DIR="${ROOT_DIR}" \
  PYTHON="${ROOT_DIR}/envs/tau2/bin/python" \
  TAU2_DOWNLOAD_AREAL=1 \
  "${ROOT_DIR}/scripts/prepare_data.sh" \
  --data tau2
```

If a public endpoint is unavailable, stage an equivalent source directory and
set `APPWORLD_DATA_SOURCE_DIR`, `WEBSHOP_DATA_SOURCE_DIR`,
`TAU2_DATA_SOURCE_DIR`, or `TAU2_AREAL_SOURCE_DIR`. Those are generic inputs;
they must not be hard-coded to one organization's cloud mount.

Private MCP data and teacher traces cannot be reconstructed from public
sources. Transfer them as separate access-controlled assets, validate them,
and never claim the public cold-start flow recreated them.

## Build And Validate Data Packs

```bash
ROOT_DIR="${ROOT_DIR}" \
  LOCAL_ENVS_DIR="${ROOT_DIR}/envs" \
  "${ROOT_DIR}/scripts/pack_data.sh" \
  --data alfworld,appworld,tau2,webshop \
  --validate-data-load
```

Test the same materialization contract used by training on a clean local path:

```bash
export LOCAL_ENVS_DIR=/local/fast-disk/server-ops-envs
export LOCAL_RUNTIME_DIR=/local/fast-disk/server-ops-runtime
rm -rf "${LOCAL_ENVS_DIR}" "${LOCAL_RUNTIME_DIR}"

ROOT_DIR="${ROOT_DIR}" \
  "${ROOT_DIR}/scripts/materialize_node_runtime.sh" \
  --envs slime,alfworld,appworld,tau2,webshop,wandb \
  --data alfworld,appworld,tau2,webshop \
  --sources tau2,webshop \
  --validate-data-load
```

Do not start a formal run until materialization and the environment-specific
data validators pass on every selected node.

## Back Up The Published Bundles

Keep build prefixes and caches disposable. Back up the published bundles and
the source commit, not the expanded conda directories.

For any mounted durable filesystem:

```bash
BACKUP_ROOT=/any/mounted/durable-storage/agentic-slime-packs/$(date -u +%Y%m%d)
mkdir -p "${BACKUP_ROOT}"
rsync -a --checksum "${ROOT_DIR}/packs/" "${BACKUP_ROOT}/"
git -C "${ROOT_DIR}/code/slime" rev-parse HEAD > "${BACKUP_ROOT}/agentic-slime.commit"
```

For an object store supported by `rclone`:

```bash
rclone copy "${ROOT_DIR}/packs" \
  remote:agentic-slime-packs/$(date -u +%Y%m%d) \
  --checksum --progress
```

These are examples, not a storage-provider requirement. Use the cloud drive,
object store, or shared filesystem available in the new environment. Preserve
all four members for every runtime/data bundle. Keep credentials outside the
backup directory.

After upload, verify from a fresh download or mount:

```bash
PACK_DIR="${BACKUP_ROOT}" ROOT_DIR="${ROOT_DIR}" \
  "${ROOT_DIR}/scripts/verify_pack_bundle.sh" \
  --runtime slime,alfworld,appworld,tau2,webshop,wandb \
  --no-extract
```

## Restore On Another Cluster

1. Clone and checkout the recorded agentic-slime commit.
2. Configure the new arbitrary `ROOT_DIR` and install the skill scripts.
3. Copy the complete bundle directory from durable storage into
   `${ROOT_DIR}/packs`.
4. Run `verify_pack_bundle.sh --no-extract` before using it.
5. Materialize env/data packs to node-local paths with
   `prepare_node_runtime.sh` or `materialize_node_runtime.sh`.
6. Run data-load smoke tests, then launch only from resolved profiles.

Do not rewrite manifests after transfer. If checksum validation fails, restore
the artifact again or rebuild it; do not bless a corrupted tarball with a new
checksum.

## Definition Of Done

A cold-start handoff is complete only when:

- the exact agentic-slime commit is pushed and recorded;
- selected runtime packs have tarball, checksum, revision, and manifest;
- selected data packs have the equivalent four files;
- clean extraction/import and environment data-load smokes pass;
- the complete bundle is copied to durable storage available to the next
  cluster/operator;
- private datasets, models, credentials, and experiment checkpoints are listed
  separately with their own access and restore instructions.
