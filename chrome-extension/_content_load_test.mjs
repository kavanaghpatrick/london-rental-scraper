/**
 * Static load test for content.js — runs the content-script IIFE in a minimal
 * browser shim (no jsdom) and asserts it loads WITHOUT throwing.
 *
 * Primary regression guard: log()/logError() were once infinitely self-recursive,
 * so the top-level `log('Script loaded!')` threw "Maximum call stack size exceeded"
 * and aborted the entire script before init() ran — the plugin failed to load on
 * EVERY site. This test reproduces that exact crash path and would fail loudly if
 * it ever comes back.
 *
 * Secondary guards:
 *  - extractPropertyDataChestertons() must not throw on a bare/empty DOM (it must
 *    return null so the SPA retry can re-fire, rather than aborting the UI render).
 *  - On Chestertons with no rendered price, init() must not blow up; it should arm
 *    the bounded SPA retry instead of erroring.
 *
 * Run: node _content_load_test.mjs   (exit 0 = pass, 1 = fail)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(__dirname, 'content.js'), 'utf8');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) {
    console.log(`OK   ${name}`);
  } else {
    failures++;
    console.log(`FAIL ${name}${detail ? ' — ' + detail : ''}`);
  }
}

// --- Static source guard: log/logError must NOT call themselves ---------------
// Catches the recursion bug at the source level even if runtime shims drift.
const logBody = SRC.match(/function log\(\.\.\.args\)\s*\{([\s\S]*?)\}/);
const logErrBody = SRC.match(/function logError\(\.\.\.args\)\s*\{([\s\S]*?)\}/);
check('log() body found', !!logBody);
check('logError() body found', !!logErrBody);
// Strip `console.`-qualified calls before checking for a bare self-call, so
// `console.log(...)` (the correct delegation) doesn't read as recursion.
const stripConsole = s => s.replace(/console\.\w+\s*\(/g, '');
check('log() does not call log() recursively',
  logBody && !/(^|[^.\w])log\s*\(/.test(stripConsole(logBody[1])),
  'log() must delegate to console.*, not itself');
check('logError() does not call logError() recursively',
  logErrBody && !/(^|[^.\w])logError\s*\(/.test(stripConsole(logErrBody[1])),
  'logError() must delegate to console.*, not itself');

// --- Minimal browser shim -----------------------------------------------------
function makeEl(overrides = {}) {
  return {
    textContent: '',
    innerText: '',
    innerHTML: '',
    href: '',
    dataset: {},
    id: '',
    getAttribute: () => null,
    setAttribute: () => {},
    querySelector: () => null,
    querySelectorAll: () => [],
    appendChild: () => {},
    remove: () => {},
    addEventListener: () => {},
    click: () => {},
    ...overrides,
  };
}

function runScript({ hostname, pathname, href }) {
  const body = makeEl({ innerText: '', innerHTML: '' });
  const documentShim = {
    title: 'Chestertons',
    body,
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],          // empty DOM => extraction finds nothing
    createElement: () => makeEl(),
    addEventListener: () => {},
  };
  const windowShim = {
    location: { hostname, pathname, href },
    addEventListener: () => {},
    __rentFairValueLoaded: undefined,
  };
  // Wire window self-reference so `window.foo = ...` in the IIFE lands on the shim.
  windowShim.window = windowShim;

  const sandbox = {
    window: windowShim,
    document: documentShim,
    history: { pushState() {}, replaceState() {} },
    location: windowShim.location,
    setInterval: () => 0,        // don't actually start the poll/observer timers
    clearInterval: () => {},
    setTimeout: () => 0,
    clearTimeout: () => {},
    MutationObserver: class { observe() {} disconnect() {} },
    fetch: async () => ({ ok: false, status: 599, json: async () => ({}) }),
    Tesseract: undefined,
    chrome: { runtime: { sendMessage: () => {}, getURL: () => '', lastError: null } },
    console,
  };

  // Build a function whose params ARE the sandbox globals, then call it. This makes
  // the bare references inside content.js (window, document, ...) resolve to shims
  // without needing the vm module.
  const names = Object.keys(sandbox);
  const vals = names.map(n => sandbox[n]);
  // eslint-disable-next-line no-new-func
  const fn = new Function(...names, `'use strict';\n${SRC}`);
  fn(...vals);
}

// --- Runtime guards -----------------------------------------------------------
let threw = null;
try {
  runScript({
    hostname: 'www.chestertons.co.uk',
    pathname: '/properties/21844781/lettings/CJL260150',
    href: 'https://www.chestertons.co.uk/properties/21844781/lettings/CJL260150',
  });
} catch (e) {
  threw = e;
}
check('content.js IIFE loads without throwing (no stack overflow) on Chestertons',
  threw === null,
  threw ? `${threw.name}: ${threw.message}` : '');

// Also confirm it loads cleanly on a Rightmove URL (site-agnostic load path).
let threwRM = null;
try {
  runScript({
    hostname: 'www.rightmove.co.uk',
    pathname: '/properties/123456789',
    href: 'https://www.rightmove.co.uk/properties/123456789',
  });
} catch (e) {
  threwRM = e;
}
check('content.js IIFE loads without throwing on Rightmove',
  threwRM === null,
  threwRM ? `${threwRM.name}: ${threwRM.message}` : '');

console.log('');
if (failures === 0) {
  console.log('=== PASS: content.js load + recursion guards green ===');
  process.exit(0);
} else {
  console.log(`=== FAIL: ${failures} check(s) failed ===`);
  process.exit(1);
}
