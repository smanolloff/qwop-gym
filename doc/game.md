# Game

Here you will find details about the `QWOP` game.

## Launching the game (pure browser mode)

To start (or debug) the game in standalone mode (without the RL
env), you must load [QWOP.html](./game/QWOP.html) in a web browser
(type `file:///path/to/QWOP.html` in the address bar).

Browser security restrictions may prevent the game from loading - you will
need to find out how to bypass them. For Chromium-based browsers (Chrome, Brave, etc.)
you can do it by passing the
`--allow-file-access-from-files` and `--allow-cross-origin-auth-prompt`
[command-line options](https://peter.sh/experiments/chromium-command-line-switches/)
to the binary, for example:

```bash
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
    --allow-file-access-from-files \
    --allow-cross-origin-auth-prompt \
    --user-data-dir=$(mktemp -d) \
    'file:///path/to/QWOP.html'
```

Except for a few warnings related to the AudioContext, there should be only one
error in the browser console - a WebSocket error which you can ignore.

> [!NOTE]  
> Since the game has been modified to allow control from an RL environment, it
> is not "playable": it runs in a step-by-step manner, one frame at a time,
> only when a "step" command is given.

## Keyboard controls

| key | effect |
|-----|--------|
| QWOP | standard move commands |
| R | restart game |
| S | step (advance one timestep) |
| D | draw (render) |
| F | step, then draw |

## Configuration

Passing the below options as url parameters will control various game aspects:

| option | type | default | description |
|--------|------|---------|-------------|
| `port` | int | | WebSocket port to communicate with the RL env |
| `seed` | int | | A seed for the game's random generator |
| `intro` | bool | `true` | Enables the game intro screen |
| `stepsize` | int | `1` | Number of updates per step |
| `text` | string | | Static text to display |
| `stat` | bool | `false` | Display a table with various env stats |
| `game` | bool | `true` | Enables the game canvas itself * |

\* Hiding the game is used during RL training, where no rendering occurs anyway

## About the QWOP source code

Unfortunately, QWOP's source code is not officially published and can't be
distributed as a part of this project.

However, the source code is available
[here](https://www.foddy.net/QWOP.min.js), so you can download and apply the
needed modifications with this simple command:

```bash
curl -sL https://www.foddy.net/QWOP.min.js | python src/game/patcher.py
```

The above command will download the original source code and save a _modified_
version of it to `src/game/QWOP.min.js` (which is what you need to start the env).
