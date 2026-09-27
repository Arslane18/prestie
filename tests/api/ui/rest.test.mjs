// Unit tests for the resting logic: `node --test tests/api/ui/rest.test.mjs`
import assert from "node:assert/strict";
import { test } from "node:test";

import { createRestController } from "../../../src/prestie/api/static/rest.js";

/** A fake clock: timers only fire when the test says so. */
function fakeTimers() {
  const pending = new Map();
  let next = 0;
  return {
    schedule(callback, delay) {
      next += 1;
      pending.set(next, { callback, delay });
      return next;
    },
    cancel(handle) {
      pending.delete(handle);
    },
    fireAll() {
      const due = [...pending.values()];
      pending.clear();
      for (const timer of due) timer.callback();
    },
    get delays() {
      return [...pending.values()].map((timer) => timer.delay);
    },
  };
}

function setup() {
  const timers = fakeTimers();
  const changes = [];
  const controller = createRestController({
    delayMs: 6000,
    schedule: timers.schedule,
    cancel: timers.cancel,
    onRest: () => changes.push("rest"),
    onWake: () => changes.push("wake"),
  });
  return { timers, changes, controller };
}

test("rests after the delay once the pointer is out and the game has the focus", () => {
  const { timers, changes, controller } = setup();

  controller.pointerLeave();
  controller.blur();
  assert.deepEqual(timers.delays, [6000]);
  timers.fireAll();

  assert.equal(controller.resting, true);
  assert.deepEqual(changes, ["rest"]);
});

test("never rests while the window has the focus, e.g. while typing", () => {
  const { timers, changes, controller } = setup();

  controller.pointerLeave();
  timers.fireAll();

  assert.equal(controller.resting, false);
  assert.deepEqual(changes, []);
});

test("never rests while the pointer is over the window, e.g. while reading", () => {
  const { timers, changes, controller } = setup();

  controller.blur();
  controller.pointerEnter();
  timers.fireAll();

  assert.equal(controller.resting, false);
  assert.deepEqual(changes, []);
});

test("hovering wakes the window at once and cancels a pending rest", () => {
  const { timers, changes, controller } = setup();
  controller.pointerLeave();
  controller.blur();
  timers.fireAll();

  controller.pointerEnter();

  assert.equal(controller.resting, false);
  assert.deepEqual(changes, ["rest", "wake"]);
  controller.pointerLeave();
  controller.pointerEnter();
  timers.fireAll();
  assert.deepEqual(changes, ["rest", "wake"]);
});

test("taking the focus wakes the window", () => {
  const { timers, changes, controller } = setup();
  controller.pointerLeave();
  controller.blur();
  timers.fireAll();

  controller.focus();

  assert.equal(controller.resting, false);
  assert.deepEqual(changes, ["rest", "wake"]);
});

test("rests only once however many times the conditions repeat", () => {
  const { timers, changes, controller } = setup();

  controller.pointerLeave();
  controller.blur();
  controller.blur();
  timers.fireAll();
  controller.pointerLeave();
  timers.fireAll();

  assert.deepEqual(changes, ["rest"]);
  assert.deepEqual(timers.delays, []);
});
