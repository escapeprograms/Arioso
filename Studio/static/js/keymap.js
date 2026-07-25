// keymap.js — window keydown -> action dispatch. Actions are injected by main.js so this
// module stays pure/testable (dispatch() can be called directly with a plain object).
let actions = null;
export function init(a){ actions = a; }

function inField(el){ return el && /^(input|select|textarea)$/i.test(el.tagName); }

export function onKeyDown(e){
  if (inField(document.activeElement)) return;
  if (dispatch(e)) e.preventDefault();
}

// Returns true if handled. `e` may be a real KeyboardEvent or {key,ctrlKey,shiftKey,metaKey}.
export function dispatch(e){
  const k = e.key;
  const ctrl = e.ctrlKey || e.metaKey;
  const shift = e.shiftKey;
  if (!actions) return false;

  // Arrow keys nudge the selection: Left/Right by one grid step (Shift = fine, 1/4 step),
  // Up/Down by a semitone (Ctrl = octave). Ctrl+Left/Right stay unbound. This runs before
  // the Ctrl letter switch below, whose `default: return false` would otherwise swallow
  // Ctrl+arrows. Nudge actions return false on an empty selection so the browser default runs.
  switch (k){
    case 'ArrowLeft':  return ctrl ? false : actions.nudgeTime(-1, shift);
    case 'ArrowRight': return ctrl ? false : actions.nudgeTime(+1, shift);
    case 'ArrowUp':    return actions.nudgePitch(+1, ctrl);
    case 'ArrowDown':  return actions.nudgePitch(-1, ctrl);
  }

  if (ctrl){
    switch (k.toLowerCase()){
      case 'z': shift ? actions.redo() : actions.undo(); return true;
      case 'y': actions.redo(); return true;
      case 'c': actions.copy(); return true;
      case 'x': actions.cut(); return true;
      case 'v': actions.paste(); return true;
      case 'a': actions.selectAll(); return true;
      case 's': actions.save(); return true;
      case 'r': if (actions.render){ actions.render(); return true; } return false;  // overrides browser reload
      default: return false;
    }
  }

  // Shift+1..4 -> set articulation on the selection (digits 1..4 alone switch tools, so
  // techniques take the shifted variant). Uses e.code because Shift changes e.key.
  if (shift && !ctrl && /^Digit[1-4]$/.test(e.code || '')){
    actions.setTechnique(e.code.slice(-1)); return true;
  }

  switch (k){
    case ' ': case 'Spacebar': actions.playPause(); return true;
    case 'Delete': case 'Backspace': actions.deleteSelected(); return true;
    case 'Escape': actions.deselect(); return true;
    // FL-style tool letters
    case 'p': case 'P': actions.setTool('draw'); return true;
    case 'e': case 'E': actions.setTool('select'); return true;
    case 'd': case 'D': actions.setTool('delete'); return true;
    case 'c': case 'C': actions.setTool('slice'); return true;
    case '1': actions.setTool('draw'); return true;
    case '2': actions.setTool('select'); return true;
    case '3': actions.setTool('delete'); return true;
    case '4': actions.setTool('slice'); return true;
    case 'l': case 'L': actions.toggleLoop(); return true;
    case 'f': case 'F': actions.toggleFollow(); return true;
    case 'F9': if (actions.render){ actions.render(); return true; } return false;
    default: return false;
  }
}
