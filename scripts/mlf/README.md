# MLF runtime and pack scripts

These scripts keep code, reusable packs, and node-local runtime state separate.

- NAS root: `/mnt/bn/jixf-nas-lq/mlf`
- reusable packs: `${MLF_NAS_ROOT}/packs`
- reusable source/data/model backups: `${MLF_NAS_ROOT}/{code,data,models}`
- node-local envs: `/tmp/mlf-envs`
- node-local source/data/model runtime: `/tmp/mlf-runtime`

Normal node migration is:

```bash
bash scripts/mlf/publish_slime_pack.sh
bash scripts/mlf/build_webshop_env.sh
bash scripts/mlf/build_alfworld_env.sh
bash scripts/mlf/materialize_node_runtime.sh
```

The env packs are intentionally independent:

- `slime.tar.gz`: training, Ray, SGLang, Megatron, torch, CUDA toolkit.
- `webshop.tar.gz`: WebShop HTTP environment server dependencies.
- `alfworld.tar.gz`: ALFWorld/TextWorld HTTP environment server dependencies.

The adapter code in `examples/` should consume these packs. It should not install
Python packages during normal training startup.
