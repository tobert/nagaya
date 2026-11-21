# The page of the web app

`tools/nagaya_web.py` serves this folder. `index.html` is the page, and the app serves each `.css` and `.js` file
here under `/app/web/`. The app reads a file at each request, so a change shows at a reload. The page needs no build
step.

## The files

| File | Holds |
|---|---|
| `index.html` | the rail of tabs, the bar, and one `<section>` for each tab |
| `style.css` | the tokens, the shell, the shared parts, then the parts of one tab |
| `core.js` | the nagaya on show, the calls, the served models, the objects, the keep and the tabs |
| `engines.js` | the Engines tab, and `loadEngine` and `ready`, which each tab uses |
| `chat.js` | the Chat tab: each served model whose replies hold text, with the controls that its identity declares |
| `images.js`, `speech.js`, `embed.js`, `lab.js`, `steer.js` | one tab each |
| `main.js` | the start: the first tab, the keep and the refresh loop |

The scripts are classic scripts, not modules. They share one scope, and they load in the order of `index.html`.
`steer.js` loads after the other tabs, because it fills the vector list of each of them.

## The controls of the Chat tab

The Chat tab lists each served model whose replies hold text. A control shows only for a model that declares it in
its identity (gogi "Identity, limits, capabilities"):

| Control | The model declares | The context gets |
|---|---|---|
| Thinking | the capability `thinking` | `thinking: true` when the box is on |
| Attach | a type in `media.inputs` | an object part for each file of the message |
| Reply | an `audio/*` type in `media.outputs` | `output` with the mode, or a TTS request for each chunk |
| Steering | the capability `control_vectors` | `control` with the vector of the model |
| Layer loop | the capability `layer_loops` | `loop`, with a start and an end from `bounds` |
| Cycles | the capability `cycles` | `cycles`, when the field is set and is not the `default` |

A turn after a reply names the record of the reply in `parent`, and it sends only its new message. Its settings
stand beside the parent. `loop: null` and `cycles: null` drop a loop or a count that the record holds. A 404 that
names `context.parent` makes the page send the chat whole.

## Add a tab

1. Add a button with `data-tab="<id>"` to a group of the rail in `index.html`.
2. Add `<section id="<id>">` with a `.head`, and build it from the shared parts of `style.css`: `.cols`, `.card`,
   `.row`, `.fields`, `.model`, `.actions`, `.composer`, `.note` and `.err`.
3. Add `<id>.js`, and name it in `index.html` before `steer.js`.
4. Add tests to `tools/test_nagaya_web_page.py`.

The core gives a tab these hooks:

- `modelSelect(id, accepts)` makes `<select id>` the model list of the tab. It needs a `<span id="<id>-state">`.
- `onRefresh(draw)` runs `draw` after each read of the served models, each 3 s.
- `onHost(draw)` runs `draw` after the choice of another nagaya.
- `keeps(list)` adds the objects of the tab, each `{id, host}`, to the keep of the page.
- `recallJson(key, otherwise)` and `rememberJson(key, value)` keep state in this browser.
- `call(method, path, body)` and `stream(body, signal, on)` send a request to the nagaya on show.
- `ready(model, status)` waits for a served model, and loads its engine when it is not loaded.

## Tests

Run them in `tools/`:

```
python3 -m unittest test_nagaya_web test_nagaya_web_page
```

`test_nagaya_web_page` runs the scripts in QuickJS with a stub page. It skips without the package `quickjs`.
