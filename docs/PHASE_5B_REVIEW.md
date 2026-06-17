# Phase 5B README Review — Hiring-Manager Simulation

## Time to Comprehension

Roughly 80-90 seconds for a first scan. The tagline at lines 3-6 ("grounds answers in verifiable Prolog logic over CLIP / OWL-ViT scene graphs ... the LLM only translates English to a query that the symbolic stack executes") gave me the one-line pitch immediately. The architecture ASCII block (lines 40-98) took the longest single chunk of attention — maybe 25-30 seconds — because I was checking whether the boxes were just shapes or whether they carried real content (they do: model names, thresholds, retry semantics). The results table at line 110 is scannable in under 5 seconds. A deeper read with the architecture deep-dive section took me into the 3-minute range.

## Comprehension Test Answers

1. **One sentence**: A visual question answering pipeline that does perception with local vision models (OWL-ViT + CLIP) into a pydantic SceneGraph, then translates an English question to a Prolog query via a local LLM and executes it on SWI-Prolog so the answer is symbolically derivable and stage-attributable.

2. **The two numbers**:
   - **100.0%** (32/32) on a hand-authored synthetic suite where vision is bypassed and the SceneGraph is fed in from a preset — this isolates the LLM + Prolog stack.
   - **56.0%** (28/50) on a 50-question CLEVR subset with the vision pipeline running end-to-end (zero-shot, no fine-tuning).

3. **Architecture (data flow)**:
   - Input image →
   - `scene_extractor`: OWL-ViT zero-shot detection + per-class NMS → CLIP attribute scoring (color/size/material/shape with a small prompt ensemble) → pure-Python geometric relations over normalized bboxes →
   - pydantic-validated `SceneGraph` →
   - `kb_generator`: emits Prolog facts (`object/2`, `attribute/3`, `relation/3`) + 16 derived rules →
   - `nl2prolog`: qwen2.5-coder:7b via ollama with 15 few-shot pairs, ≤3-attempt retry loop validated by subprocess `swipl` →
   - `query_executor`: pyswip wrapped in `call_with_time_limit/2` →
   - `verbalizer`: per-qtype templates emitting answer + reasoning trace.

4. **What makes it different from a generic RAG bot**: The LLM is firewalled to the single job of English-to-Prolog translation; the actual answer comes out of a deterministic symbolic stack. Concretely: line 36 — "given correct facts and a correct query, the answer is correct" — and the failure-bucketing taxonomy (vision / kb_validation / translation / execution / verbalization / correctness) at lines 102-104 means every wrong answer has a known stage of origin. That is a structurally different architecture than "shove context into a prompt and trust the LLM."

## Honesty Check

**Overclaiming / superlatives / SOTA claims**: I did not find any. There is no "state of the art", no "best", no "first", no "novel." The CLEVR result is labeled "zero-shot, no fine-tuning" and is explicitly 56%, not dressed up. The 100% number is immediately and explicitly qualified at line 119: "Vision is bypassed — the SceneGraph is produced directly from the preset, isolating LLM + Prolog accuracy." That is exactly the right disclosure.

**Marketing language**: None of "revolutionary / groundbreaking / cutting-edge / seamlessly / robust" appears. The prose is technical throughout — e.g. line 33-36 frames the design choice as a debuggability argument, not a marketing claim. Tone is consistently engineer-to-engineer.

**Burying the CLEVR number**: Presented honestly side-by-side. The results table at lines 110-113 puts Synthetic and CLEVR in the same table, same column widths, both bolded — the candidate did NOT put 100% in a hero banner above CLEVR. If anything the harder number (56%) gets MORE explanatory real estate (lines 138-181 vs lines 115-135), including a candid "aborted — no config beat 56% with synthetic gate intact" line at row 4.2 in the tuning table. That is the opposite of burying.

**Decorative diagrams**: The ASCII diagram is informative, not decorative. It carries actual content: model identifiers (`google/owlvit-base-patch32`, `openai/clip-vit-base-patch32`, `qwen2.5-coder:7b`), thresholds (`threshold 0.1`), runtime targets (`on MPS`), design choices (`≤ 3-attempt retry loop, error fed back to model`, `subprocess swipl validator`, `call_with_time_limit/2 for in-Prolog timeout`), the predicate signatures (`object/2, attribute/3, relation/3`), and the 16-rule count. A reader who skipped the prose would still come away with concrete design knowledge. This is not ASCII-art-for-the-look.

**Factual inconsistencies**: I cross-checked and found none material. The architecture says qwen2.5-coder:7b (line 79); the results section confirms qwen2.5-coder:7b is "the model variant that gets to 100%" (line 130); the tech stack repeats it (line 308). The 15 few-shot pairs claim (line 79) matches the deep-dive (line 252). The 16 derived rules claim (line 71) matches the deep-dive (lines 239-240). The harness's six-stage failure taxonomy (line 103) is repeated consistently (line 271). The one place to watch is line 134, which mentions "the original 30-question subset" for the llama baseline while the current suite is 32 — but this is explained as a baseline-vs-current distinction, not a contradiction, and the file paths for both runs are named.

**Rehearsed "what I learned"**: This reads like an engineer, not a hiring-manager-target. The three bullets at lines 279-297 are technically specific surprises:
- "The text encoder's prior for 'a photo of a metal object' pattern-matches to pots and pans, not chrome spheres" — that is a real, specific, debugged observation, not "I learned to ship."
- "llama3.2:3b dropped `findall+length` wrappers on count questions about half the time, even with the few-shot pair in the prompt" — that is a specific empirical failure mode, with a named workaround (qwen2.5-coder:7b) and stated costs (~2 GB disk, ~2 s per query).
- "Detection-prompt rephrasing made it worse, not better; the prior shift trades one set of misses for another" — an admission, not a brag.

No "I learned the importance of collaboration." No "I learned to iterate quickly." This sounds like the author has actually been in the code.

## Concerns

A few things I'd flag, none of them disqualifying:

- **"Live demo: coming soon"** at line 8 is the weakest line in the document. For a portfolio piece, "coming soon" is the universal tell that the link is dead. Either ship the Spaces demo (line 217 implies it exists in synthetic-only mode), or drop the "coming soon" line entirely and just link to local-run instructions. Saying "coming soon" without a date erodes trust slightly.
- The CLEVR subset size is small — 50 questions over 10 scenes. The candidate is honest about this and the per-qtype breakdown (lines 144-147) shows the cases are concentrated on count (30) and boolean (20). A skeptical reader might want to see 200-500 cases to feel the number is stable. This is a real limitation but the README does not hide it.
- The CI badge at line 10 is good signaling, but no test count appears in the README until line 194 ("132 passed, 11 deselected in ~3s") inside the quick-start block. A junior portfolio could benefit from putting that number near the top.
- Line 219 — "Spaces deployment runs synthetic-only mode (vision pipeline is too heavy for the Spaces free CPU tier)" — is honest but mildly self-undercutting given that vision is the half doing the heavy lifting and is where the 56% number comes from. Worth keeping, but it's a tension worth being aware of.
- Minor: the "Tuning history" table at lines 172-176 has two "aborted" rows. They are honest, but a reader could come away thinking "this is stuck." The prose immediately after ("the obvious next step is fine-tuning the detector on a CLEVR slice") is the right counter, but a sharper one-line "what's next" inside the table cell might help.

Nothing in the README struck me as amateurish in a way that would make me close the tab. The prose is tight, the numbers are honest, and the architecture is communicated at the level of "I could re-implement this from the README and a weekend." For a sophomore, this is well above the bar.

## Verdict

The candidate has built a real system, measured it honestly, attributed the failures correctly, and written it up like an engineer rather than a salesperson. The 56% CLEVR number is presented alongside (not under) the 100% synthetic number, with explicit framing that the synthetic suite bypasses vision. The architecture diagram carries information, not aesthetics. The "what I learned" section names specific failure modes (chrome-sphere recall, `findall+length` dropouts) instead of generic platitudes. I would keep reading and likely clone the repo.

Verdict: PASS
