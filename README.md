# claude-opus-4-8: literal `<cite>` XML tags leak instead of structured citations

After switching our product over to claude-opus-4-8, we noticed it randomly
leaking literal `<cite>` tags into response text where we'd normally receive
structured citations. The conditions under which it happens aren't 100% clear
to us — it's stochastic, and our best theory of the contributing factors is
below. We've also observed noticeable back-pressure on the API's SSE stream
(sometimes waiting 25-30 seconds for the next event) right before the failure
mode appears, though we're unsure whether that's related. **This has not been
seen in any of the previous models.** This repo represents our best attempt at
a minimum reproducible example.

When given `search_result` blocks (with `citations: {"enabled": true}`) in a
tool_result, `claude-opus-4-8` intermittently returns its citations as
**literal XML inside the response text**, with `citations: null`:

```json
{
  "type": "text",
  "text": "... <cite index=\"1-1\">The internal investigator interviewed the nurse and treating physician, Alex Karev and Dr. Arizona Robbins, respectively.</cite> ...",
  "citations": null
}
```

instead of the expected structured form:

```json
{
  "type": "text",
  "text": "...The internal investigator interviewed the nurse...",
  "citations": [
    {
      "type": "search_result_location",
      "source": "https://foo.bar/files/internal-investigation-findings.pdf",
      "title": "Internal Investigation Findings",
      "cited_text": "..."
    }
  ]
}
```

Failures are usually all-or-nothing per response (~20 literal tags, zero
structured citations) though partial leaks (1-2 tags in an otherwise clean
response) were occasionally observed. The `<cite index="X-Y">` notation
strongly resembles an internal citation format, suggesting the model emits it
natively and the step that converts it into structured citations is bypassed.

**claude-opus-4-7 and claude-sonnet-4-6 never do this** on byte-identical
requests.

## What seems to trigger it (our best theory, not certainty)

We arrived at this request shape by trial and error. The failure is
stochastic and our sample sizes are modest, so we can't claim a definitive
cause — but some combination of the following five things appears to put
claude-opus-4-8 into the failing mode, and in our testing, simplifying any
one of them made the failures stop or become much rarer:

1. **A prior assistant tool_use whose input talks about quoting and
   citing.** The conversation contains the model's own earlier delegation
   text, e.g. "capture a verbatim or near-verbatim supporting quote so I
   can cite it precisely." A neutral task in the same slot (arm
   `no-cite-task`) didn't fail for us.

2. **Long, detailed tool descriptions.** The repro includes only five tools,
   but their descriptions total ~5K characters. The same tools with much
   shorter descriptions (arm `small-tools`) didn't fail for us.

3. **Tool descriptions that talk about citations and locations as data** —
   e.g. "citation: Links between facts and sources", "page 1 = location
   \"0\"" — written in forceful MUST/MANDATORY style. Calmer rewrites
   without that framing seemed to fail less often.

4. **Tool descriptions using the same vocabulary as the question and the
   search results** (source / fact / case). The same descriptions reworded
   into an unrelated domain didn't fail for us.

5. **Overlapping search results** — each document appears twice, as a full
   page and as a sub-chunk of the same page, so the same passage exists
   under two result indices. Payloads without this overlap failed much
   less often.

A broad user question that asks for an exhaustive cited list ("list all the
facts with citations") also appears to matter; narrower questions failed
less often for us.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env           # put your ANTHROPIC_API_KEY in .env
python reproduce.py            # 3 models x 3 arms x 12 attempts (~108 requests)
```

Quicker / narrower:

```bash
python reproduce.py --attempts 6
python reproduce.py --models claude-opus-4-8 --arms trigger
```

The script prints each attempt and a final failure-rate table. An attempt
counts as a failure only when the response text contains literal `<cite`
tags — nothing else affects the rate. Any other XML-like tags, if ever seen,
are flagged informationally but never counted as failures (we have only ever
observed `<cite>`).

## Results we observe (2026-06-06, streaming API, one run of `python reproduce.py`)

| arm | meaning |
|---|---|
| `trigger` | the full request shape described above (all five factors) |
| `no-cite-task` | same, but the prior tool_use task has no quote/cite language |
| `small-tools` | same, but with the much shorter tool descriptions |

| model | arm | failures | rate |
|---|---|---|---|
| **claude-opus-4-8** | **trigger** | **6/12** | **25-50% across runs** |
| claude-opus-4-8 | no-cite-task | 0/12 | 0% |
| claude-opus-4-8 | small-tools | 0/12 | 0% |
| claude-opus-4-7 | trigger | 0/12 | 0% |
| claude-opus-4-7 | no-cite-task | 0/12 | 0% |
| claude-opus-4-7 | small-tools | 0/12 | 0% |
| claude-sonnet-4-6 | trigger | 0/12 | 0% |
| claude-sonnet-4-6 | no-cite-task | 0/12 | 0% |
| claude-sonnet-4-6 | small-tools | 0/12 | 0% |

The trigger rate is stochastic and varies between runs (observed 25-50%
across sessions); the ablation arms and the other models have stayed at 0%
across hundreds of attempts.

## Files

| File | Contents |
|---|---|
| `reproduce.py` | The matrix runner (single file, stdlib + `anthropic`) |
| `data.json` | System prompt, user question, both task strings, the explore tool definition, and the 6 `search_result` blocks (3 documents, each as a full page plus an overlapping sub-chunk; plain URL sources) |
| `tools_large.json` | 4 synthetic tool definitions (~5K chars) carrying ingredients 3+4 |
| `tools_small.json` | The same 4 tools at ~2K chars — flips arm `small-tools` clean |
