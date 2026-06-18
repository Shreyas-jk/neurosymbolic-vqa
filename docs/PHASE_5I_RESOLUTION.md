# Phase 5i — Resolution: HF-only LFS migrate + force-push

The cached CLEVR mode is now live on both remotes. This doc records why
GitHub's and HuggingFace's git histories are now intentionally divergent.

## Why HF rejected the original push

The Phase 5i task added five raw PNG files (~190 KB each) under
`demo/cached_scenes/`. HuggingFace's pre-receive hook rejects any commit
in the push pack that introduces a binary blob outside their
[Xet/LFS storage layer](https://huggingface.co/docs/hub/xet):

```
remote: Your push was rejected because it contains binary files.
remote: Please use https://huggingface.co/docs/hub/xet to store binary files.
remote: Offending files:
remote:   - demo/cached_scenes/CLEVR_val_000000.png (ref: refs/heads/main)
remote:   ... (5 PNGs total)
```

A follow-up commit that added `.gitattributes` LFS rules (commit
`24a3399` on GitHub's history) didn't help — HF scans the whole pack,
and the earlier commit `92bce3e` already contained the raw 183144-byte
blobs. The hook fires on any commit in the push range, not just the tip.

## Why force-pushing to HF (only) was acceptable

The standing rule "never force-push to main" exists because force-pushes
destroy reviewable history on the canonical code repo, which is GitHub.
HuggingFace Space remotes are a different role:

| Remote        | Purpose                          | History matters?     |
|---------------|----------------------------------|----------------------|
| `origin` (GitHub)   | Canonical code / portfolio review | **Yes** — protected against force-push |
| `huggingface` | Deployment target for the Gradio Space | No — only the tip matters |

No human inspects HF git history; the platform reads the tip commit to
build the Space. Rewriting it once to move blobs into LFS has the same
semantic outcome as never having added the raw blobs in the first place.

The user explicitly approved this exception when resuming Phase 5i.

## What the migrate did

```bash
git lfs migrate import --include="*.png" --everything
```

`--include="*.png"` scopes the rewrite to PNG blobs only.
`--everything` walks every ref/commit reachable, replacing the raw blobs
with LFS pointers in-place. Local commit SHAs that touched a PNG (or
sat on the same history line as one) got new SHAs propagated forward —
30 commits in total.

Pre-migrate tip: `5b6023a`. Post-migrate tip: `049da74`.
Local main was then force-pushed to `huggingface main`. The 5 LFS
objects (953 KB total) were uploaded to HF's Xet storage as part of the
push.

## GitHub vs HF history divergence

After the force-push:

| Remote        | tip commit | history shape                          |
|---------------|------------|----------------------------------------|
| `origin/main` | `57014cc`  | raw PNGs in `92bce3e`; everything else inline |
| `huggingface/main` | `d2cc550`  | LFS pointers from migrate import; 30 commits rewritten |

Both remotes contain the same working-tree content. Only the storage
layer for the PNGs and the SHA chain differ. The cached CLEVR mode is
live on both.

`git log origin/main..huggingface/main` shows 30 commits each way — no
common recent ancestor. This is permanent. Future commits to either
remote will need to be pushed via the side-branch pattern below.

## Pushing to both remotes from now on

The two history lines never meet again. For any future commit that
needs to land on both remotes, the workflow is:

```bash
# 1. Commit locally on `main` (tracks the HF/LFS history)
git add <files>
git commit -m "..."
git push huggingface main          # fast-forward from d2cc550 onwards

# 2. Replay onto origin/main's history line
git fetch origin
git checkout -b _tmp origin/main
git rm BLOCKED.md      # or whatever the diff was — easiest to cherry-pick
git cherry-pick main   # apply the same commit(s) onto origin's line
git push origin _tmp:main
git checkout main
git branch -D _tmp
```

For the BLOCKED.md removal (commit `d2cc550` on HF, `57014cc` on GitHub)
and this resolution doc, the side-branch dance was followed exactly.

## Adding more cached PNGs in the future

No re-migration is needed. `.gitattributes` already routes
`demo/cached_scenes/*.png` through LFS on both remotes. Adding a new
PNG goes through LFS automatically:

```bash
.venv/bin/python scripts/precompute_clevr_scenes.py   # regen as needed
git add demo/cached_scenes/CLEVR_val_NNNNNN.png       # → LFS pointer
git add demo/cached_scenes/CLEVR_val_NNNNNN.json
git commit -m "Add cached CLEVR scene for ..."
git push huggingface main                              # pushes LFS object + pointer
```

For GitHub, the same side-branch pattern from above applies — `origin/main`
is on the raw-PNG history line; cherry-picking the new PNG commit onto
that line will require the PNG to either be a raw blob (GitHub accepts
either) or an LFS pointer (need `.gitattributes` already in place there,
which it isn't on `57014cc`). Easiest: just commit the PNG as a raw
blob to origin's line; GitHub doesn't care.

If maintaining the dual-line dance becomes painful, the cleanest
permanent fix is a one-time force-push to origin to align it with HF.
That's a policy decision; not done here.
