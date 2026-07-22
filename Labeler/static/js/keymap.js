// keymap.js — window keydown -> action dispatch. Actions are injected by main.js so
// this module is pure/testable (selftest calls dispatch() directly with a fake event).
let actions = null;

export function init(a){ actions = a; }

function inField(el){ return el && /^(input|select|textarea)$/i.test(el.tagName); }

export function onKeyDown(e){
  if (inField(document.activeElement)) return;
  if (dispatch(e)) e.preventDefault();
}

// Returns true if the key was handled. `e` may be a real KeyboardEvent or a plain
// {key, ctrlKey, shiftKey, metaKey} object (used by the selftest).
export function dispatch(e){
  const k = e.key;
  const ctrl = e.ctrlKey || e.metaKey;
  const shift = e.shiftKey;
  if (!actions) return false;

  if (ctrl){
    switch (k.toLowerCase()){
      case 'z': shift ? actions.redo() : actions.undo(); return true;
      case 'y': actions.redo(); return true;
      case 's': actions.save(); return true;
      default: return false;
    }
  }

  if (k >= '1' && k <= '9'){ actions.setTechniqueByIndex(+k - 1); return true; }

  switch (k){
    case ' ': case 'Spacebar': actions.playPause(); return true;
    case 'v': case 'V': actions.toggleVibrato(); return true;
    case 'ArrowLeft': actions.selectPrev(); return true;
    case 'ArrowRight': actions.selectNext(); return true;
    case 'ArrowUp': shift ? actions.velUp() : actions.pitchUp(); return true;
    case 'ArrowDown': shift ? actions.velDown() : actions.pitchDown(); return true;
    case '[': actions.speedDown(); return true;
    case ']': actions.speedUp(); return true;
    case 'm': case 'M': actions.toggleMuteOrig(); return true;
    case 'n': case 'N': actions.toggleMuteMidi(); return true;
    case 'f': case 'F': actions.toggleFollow(); return true;
    case 's': case 'S': actions.save(); return true;
    case 'x': case 'X': actions.splitAtPlayhead(); return true;
    case 'c': case 'C': actions.splitAllAtPlayhead(); return true;
    case 'g': case 'G': actions.togglePaintRegion(); return true;
    case 'Delete': case 'Backspace': actions.deleteSelected(); return true;
    case 'Escape': actions.deselect(); return true;
    default: return false;
  }
}
