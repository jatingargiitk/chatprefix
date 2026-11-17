# chatprefix

Check a Hugging Face chat template for the ways it can quietly disagree with itself.

Chat templates are Jinja programs, and a lot of code assumes things about them that are often false:

- **Earlier turns render the same when the chat grows.** Prefix/KV-cache reuse in multi-turn serving and
  multi-turn training data both assume that `render(messages[:k])` is a prefix of `render(messages)`.
  Templates that strip `<think>` blocks from history, close the last turn differently, or append an EOS to
  every render break that.
- **`add_generation_prompt=True` is how the template writes an assistant turn.** If the prompt you generate
  from ends in `<think>\n` but the template writes the finished turn without it, what the model sees at
  inference is not what it was fine-tuned on.
- **The system message, `tools=`, BOS and stop tokens are used.** Some templates drop the system message,
  ignore tools without an error, write a BOS that the tokenizer then adds again, or end an assistant turn with
  something that isn't a stop token.

`chatprefix` renders a short test conversation (or yours) the way `transformers`' `apply_chat_template`
does and reports each of those as `ok`, `INFO`, `WARN` or `FAIL`, with the exact characters where two
renders diverge.

## Install

```bash
pip install git+https://github.com/jatingargiitk/chatprefix
```

Python 3.9+. Depends on `jinja2` and `tokenizers`; it does not need `transformers` or `torch`.

## Usage

```bash
chatprefix Qwen/Qwen3-0.6B                 # a Hub repo (downloads only the tokenizer files)
chatprefix ./my-finetune                    # a local directory with tokenizer_config.json etc.
chatprefix template.jinja --bos '<s>'       # a bare template
```

Every example below was run with the command shown (pinned to a Hub commit so it stays reproducible);
`examples/readme_examples.sh` re-runs them and CI checks the output still matches.

### Qwen3 rewrites the last assistant turn

```
$ chatprefix Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
chatprefix Qwen/Qwen3-0.6B@c1899de289a0
  FAIL prefix    adding message 4 (user) rewrites the render of messages 1-3
                 first difference at char 138, after 'm_end|>\n<|im_start|>assistant\n'
                 shorter render has: '<think>\n\n</think>\n\nParis.<|im_end|>\n'
                 longer render has:  'Paris.<|im_end|>\n<|im_start|>user\nAnd of Italy?<|im_end|>\n'
  ok   genprompt add_generation_prompt=True matches the rendered turn for 2 assistant message(s)
  ok   system    system message is rendered
  ok   tools     tools= is rendered into the prompt
  ok   bos       no bos_token configured
  ok   eos       assistant turns end with a stop token ('<|im_end|>', '<|endoftext|>')
  INFO mask      no {% generation %} block: return_assistant_tokens_mask (and TRL assistant_only_loss) can't be used
1 fail, 0 warn
[exit 1]
```

Qwen3 writes an empty `<think></think>` block into the assistant turn only while it is the last message.
As soon as the user replies, that turn is re-rendered without it, so the cached prefix (or the training
sample that ended there) no longer matches. With `--reasoning`, each assistant message gets a
`reasoning_content`, and the reasoning is dropped from history the same way:

```
$ chatprefix Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --reasoning --check prefix --json
{
  "source": "Qwen/Qwen3-0.6B@c1899de289a0",
  "findings": [
    {
      "check": "prefix",
      "status": "FAIL",
      "message": "adding message 4 (user) rewrites the render of messages 1-3",
      "details": [
        "first difference at char 138, after 'm_end|>\\n<|im_start|>assistant\\n'",
        "shorter render has: '<think>\\nThe user wants the capital of France.\\n</think>\\n\\nPari...'",
        "longer render has:  'Paris.<|im_end|>\\n<|im_start|>user\\nAnd of Italy?<|im_end|>\\n'"
      ]
    }
  ],
  "fail": 1,
  "warn": 0
}
[exit 1]
```

### A template where nothing breaks

```
$ chatprefix Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
chatprefix Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf
  ok   prefix    each of 4 longer conversation(s) starts with the render of the shorter one
  ok   genprompt add_generation_prompt=True matches the rendered turn for 2 assistant message(s)
  ok   system    system message is rendered
  ok   tools     tools= is rendered into the prompt
  ok   bos       no bos_token configured
  ok   eos       assistant turns end with a stop token ('<|im_end|>', '<|endoftext|>')
  INFO mask      no {% generation %} block: return_assistant_tokens_mask (and TRL assistant_only_loss) can't be used
0 fail, 0 warn
[exit 0]
```

### DeepSeek-R1-Distill: the generation prompt opens a `<think>` block the turn never has

```
$ chatprefix deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 --check genprompt
chatprefix deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae0864d
  FAIL genprompt the generation prompt before message 3 is not how the template writes that turn
                 first difference at char 100, after 'apital of France?<｜Assistant｜>'
                 shorter render has: '<think>\n'
                 longer render has:  'Paris.<｜end▁of▁sentence｜>'
  FAIL genprompt the generation prompt before message 5 is not how the template writes that turn
                 first difference at char 159, after 'er｜>And of Italy?<｜Assistant｜>'
                 shorter render has: '<think>\n'
                 longer render has:  'Rome.<｜end▁of▁sentence｜>'
2 fail, 0 warn
[exit 1]
```

### Mistral v0.3: the system message only exists before the last user turn, and BOS is doubled

```
$ chatprefix unsloth/mistral-7b-instruct-v0.3 --revision 60b99dcbc7f853b7cd18472a607e8d045b96f03e --check system --check bos
chatprefix unsloth/mistral-7b-instruct-v0.3@60b99dcbc7f8
  WARN system    system message is silently dropped from the full conversation
                 it does appear when the conversation is cut to 2 or 4 messages, so training on whole conversations loses it
  WARN bos       double BOS: the template writes '<s>' and the tokenizer adds another
                 tokenizer(text) starts with ids [1, 1, 3]; '<s>' is id 1
                 apply_chat_template(tokenize=True) is fine (it passes add_special_tokens=False); tokenizing the rendered string yourself is not
0 fail, 2 warn
[exit 0]
```

### Llama 3.2: double BOS if you tokenize the rendered string yourself

```
$ chatprefix unsloth/Llama-3.2-1B-Instruct --revision 5a8abab4a5d6f164389b1079fb721cfab8d7126c --check bos
chatprefix unsloth/Llama-3.2-1B-Instruct@5a8abab4a5d6
  WARN bos       double BOS: the template writes '<|begin_of_text|>' and the tokenizer adds another
                 tokenizer(text) starts with ids [128000, 128000, 128006]; '<|begin_of_text|>' is id 128000
                 apply_chat_template(tokenize=True) is fine (it passes add_special_tokens=False); tokenizing the rendered string yourself is not
0 fail, 1 warn
[exit 0]
```

## Checks

| check | status if it fails | what it compares |
|---|---|---|
| `prefix` | FAIL | `render(messages[:k])` must be a prefix of `render(messages[:k+1])` for every k |
| `genprompt` | FAIL | `render(messages[:i], add_generation_prompt=True)` must be a prefix of `render(messages[:i+1])` for each assistant message i |
| `system` | WARN | the system message text appears in the full render, and the template doesn't raise on it |
| `tools` | WARN | rendering with `tools=[get_weather]` doesn't raise and contains the tool name (a `tool_use` template is used if the model has one) |
| `bos` | WARN | `tokenizer(rendered_text)` doesn't start with two BOS ids, and doesn't add a BOS the template didn't write (needs `tokenizer.json`) |
| `eos` | WARN | the text after each assistant message starts with the `eos_token` or a `generation_config.json` stop token |
| `mask` | INFO | the template has `{% generation %}` blocks, which `return_assistant_tokens_mask` and TRL's `assistant_only_loss` need |

A FAIL is not always a bug in the model: gpt-oss, for example, ends the final assistant message with
`<|return|>` and earlier ones with `<|end|>` by design, which `prefix` reports. What it does tell you is that
code assuming prefix stability (cache reuse, training on a conversation's prefixes) will see different text.

Exit code: `1` if any check FAILs (with `--strict`, WARNs too), `2` if the source can't be loaded, else `0`.

## Options

```
--revision REV        Hub branch, tag or commit (default main); commits are cached in ~/.cache/chatprefix
--conversation FILE   JSON list of messages to test with instead of the built-in 5-message chat
--reasoning           add reasoning_content to every assistant message
--check NAME          run only this check (repeatable)
--kwarg KEY=JSON      extra template variable, e.g. --kwarg enable_thinking=false
--date YYYY-MM-DD     what strftime_now() returns (for templates that print today's date)
--bos / --eos         special tokens for a bare .jinja file
--no-tokenizer        don't download tokenizer.json (skips the token-level checks)
--show                print the full render of the conversation
--json                machine-readable output
--strict              exit 1 on WARN as well
```

Gated or private repos: set `HF_TOKEN`. `HF_ENDPOINT` is honoured for mirrors.

## How faithful is the rendering?

The renderer mirrors `transformers/utils/chat_template_utils.py`: the same sandboxed Jinja environment
(`trim_blocks`, `lstrip_blocks`, `loopcontrols`, `{% generation %}`), the non-escaping `tojson`,
`raise_exception`, `strftime_now`, the `reasoning_content`/`thinking` aliasing, and dropping `None`-valued
message keys. Special tokens are read from `tokenizer_config.json` with `special_tokens_map.json` as
fallback, and `chat_template.jinja` takes precedence over the template inside `tokenizer_config.json`.

`tests/test_parity.py` renders the seven vendored templates in `tests/fixtures/templates` (Qwen3, Qwen2.5,
DeepSeek-R1-Distill, Phi-4-mini, Mistral v0.3, SmolLM3, gpt-oss) with and without tools and generation
prompt and compares the output with `transformers`' own `render_jinja_template`; CI runs it against the
latest `transformers`.

Not covered: multimodal processor templates (`chat_template.json` of image/audio processors is read, but
content parts aren't exercised), `continue_final_message`, and tool-call/tool-result messages in the test
conversation (pass your own with `--conversation`).

## Development

```bash
pip install -e '.[test]'
pytest                      # offline; test_parity runs only if transformers is installed
./examples/readme_examples.sh && git diff --exit-code examples/expected   # needs network
```

## License

MIT. The templates under `tests/fixtures/templates` are copied from the model repos listed in
`tests/fixtures/templates/SOURCES.md` and keep their own licenses (Apache-2.0 or MIT).
