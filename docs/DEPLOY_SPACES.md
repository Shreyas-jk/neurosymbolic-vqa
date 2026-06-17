# HuggingFace Spaces deployment

The repo is wired to deploy directly to a Gradio Space. The YAML frontmatter
at the top of `README.md` is what Spaces reads at build time:

```yaml
title: Neurosymbolic VQA
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: demo/app.py
pinned: false
```

`demo/app.py` detects the Spaces environment via `os.environ.get("SPACE_ID")`
and hides the real-image tab when set — the OWL-ViT + CLIP stack is too heavy
for the Spaces free CPU tier. The synthetic-scene tab works end-to-end
because it bypasses the vision pipeline (it constructs the SceneGraph
directly from `synthetic.presets`).

## One-time setup

1. Create a Space at <https://huggingface.co/new-space>.
2. Choose the **Gradio** SDK and the free CPU tier.
3. For "Files", point the Space at this GitHub repo. Alternatively:

```bash
git remote add space https://huggingface.co/spaces/<your-user>/neurosymbolic-vqa
git push space main
```

4. Configure the NL→Prolog backend. The default (`local`/ollama) won't work
   on a Space without a hosted ollama endpoint, so pick one:

   - **Option A — hosted ollama** (paid). Spin up ollama on a host of your
     choice and set `OLLAMA_HOST=https://<your-host>` in the Space's secrets.
     Leave `NL2PROLOG_BACKEND` unset (defaults to `local`).
   - **Option B — OpenAI** (paid per call). Set both
     `NL2PROLOG_BACKEND=openai` and `OPENAI_API_KEY=<your-key>` in the
     Space's secrets. The OpenAI backend wires `gpt-4o-mini` at temperature
     0.2 and `response_format={"type": "json_object"}` — both already
     handled by `nl2prolog.openai_backend.OpenAIBackend`.

5. Push or trigger a rebuild. The Space will install
   `demo/requirements.txt`, then run `python demo/app.py`.

`demo/requirements.txt` deliberately omits torch / torchvision / transformers
/ Pillow / numpy — the vision stack is skipped on Spaces because the
real-image tab is hidden. If you change your mind and want to enable
real-image inference on a CPU upgrade, add those back to
`demo/requirements.txt` (they're listed in the repo-root `requirements.txt`
for the local install).

## After deployment

The Space URL becomes the live demo. Update `README.md`'s "Live demo"
section with the URL once the Space is up.

## Troubleshooting

- **`NoBackendAvailableError` at startup.** Neither `OLLAMA_HOST` nor
  `OPENAI_API_KEY` is configured. The Run buttons will return an error message
  until a backend is set — the layout still loads.
- **`pyswip` import failures.** Already handled — `packages.txt` at the repo
  root lists `swi-prolog`, which Spaces' build runs via `apt-get install -y`.
  If you fork and trim deps, keep `packages.txt` intact or pyswip will fail
  to load.
- **Translation latency feels long.** OpenAI's `gpt-4o-mini` averages
  ~3-5s per query for this prompt; the synthetic eval baseline is ~4.5s/case.
  No tuning to be done from the Space side.
