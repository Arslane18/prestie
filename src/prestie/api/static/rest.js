// Resting logic, like WoW's chat frame: once the player is back in the game
// (pointer out of the window and focus in the game), the window fades out
// after a delay; hovering or focusing it brings it back at once.
// Pure: timers are injected so the behavior is testable without a clock.

/**
 * @param {{delayMs: number, schedule: Function, cancel: Function,
 *          onRest: Function, onWake: Function}} options
 */
export function createRestController({ delayMs, schedule, cancel, onRest, onWake }) {
  let pointerInside = false;
  let focused = true; // the window has just been opened, so it has the focus
  let resting = false;
  let timer = null;

  const stopTimer = () => {
    if (timer !== null) cancel(timer);
    timer = null;
  };

  const wake = () => {
    stopTimer();
    if (!resting) return;
    resting = false;
    onWake();
  };

  const restLater = () => {
    if (pointerInside || focused || resting || timer !== null) return;
    timer = schedule(() => {
      timer = null;
      if (pointerInside || focused || resting) return;
      resting = true;
      onRest();
    }, delayMs);
  };

  return {
    pointerEnter() {
      pointerInside = true;
      wake();
    },
    pointerLeave() {
      pointerInside = false;
      restLater();
    },
    focus() {
      focused = true;
      wake();
    },
    blur() {
      focused = false;
      restLater();
    },
    get resting() {
      return resting;
    },
  };
}
