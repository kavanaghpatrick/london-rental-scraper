/**
 * Shared, zero-dependency browser/DOM shim for the extension extractor tests
 * (rightmove_extract_test.mjs, knightfrank_extract_test.mjs, foxtons_extract_test.mjs).
 *
 * WHY a hand-rolled shim and not jsdom: this repo's existing node harnesses
 * (content_load_test.mjs, similar_properties_test.mjs) are deliberately zero-dep —
 * there is no chrome-extension/package.json and node_modules is intentionally absent
 * so nothing extra ships with the extension (Chrome rejects some node_modules names).
 * We keep that convention. The shim implements ONLY the DOM surface the three
 * extractors actually touch, parsed from a REAL captured page fixture:
 *
 *   document.getElementById('__NEXT_DATA__')              (Rightmove strat 1, Foxtons)
 *   document.querySelectorAll('script')   -> [{textContent}]   (Rightmove __PAGE_MODEL)
 *   document.querySelector(<css>)         -> element | null     (KF title/price/desc)
 *   document.querySelectorAll(<css>)      -> element[]          (KF floorplan/features)
 *   document.title                                              (KF fallback)
 *   document.body.innerHTML                                     (KF floorplan regex)
 *   element.textContent / .innerText / .innerHTML / .href / .src / .dataset / getAttribute
 *
 * The CSS support is intentionally narrow: tag names, '.class', '[attr]',
 * '[attr*="v"]' (case-insensitive when the original used the `i` flag), descendant
 * combinators ('a b'), comma selector-lists, and ':first'/generic tag fallbacks —
 * exactly the patterns the extractors pass. innerText is computed in DOCUMENT ORDER
 * so the "first £ wins" ordering the KF fixture encodes is faithfully reproduced.
 *
 * The extractors are IIFE closures inside content.js. We expose them with the same
 * export-trampoline trick similar_properties_test.mjs uses: inject an assignment to a
 * sandbox `__export` object right before the IIFE's closing `})();`, then run the
 * instrumented source via `new Function(...sandboxNames, src)`. `currentSite` is a
 * `const` derived from window.location.hostname, so we set the hostname and let the
 * script's own detectSite() pick the site — no setter needed.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(__dirname, 'content.js'), 'utf8');

// ---------------------------------------------------------------------------
// Tiny HTML parser -> element tree. Good enough for our fixtures (well-formed,
// no exotic edge cases). Produces nodes with tag/attrs/children + text nodes.
// ---------------------------------------------------------------------------
const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
  'link', 'meta', 'param', 'source', 'track', 'wbr']);
// Tags whose raw text must NOT be parsed as markup (so JSON in <script> is kept verbatim).
const RAWTEXT = new Set(['script', 'style']);

function parseHTML(html) {
  // Strip comments so commented-out markup (e.g. the fixture's provenance block)
  // never becomes real nodes. Scripts/styles are protected first.
  const root = { tag: '#root', attrs: {}, children: [], parent: null };
  let cur = root;
  let i = 0;
  const n = html.length;

  function pushText(text) {
    if (text) cur.children.push({ tag: '#text', text, parent: cur });
  }

  while (i < n) {
    if (html.startsWith('<!--', i)) {
      const end = html.indexOf('-->', i);
      i = end === -1 ? n : end + 3;
      continue;
    }
    if (html.startsWith('<!', i)) { // doctype
      const end = html.indexOf('>', i);
      i = end === -1 ? n : end + 1;
      continue;
    }
    if (html[i] === '<' && html[i + 1] === '/') { // closing tag
      const end = html.indexOf('>', i);
      const name = html.slice(i + 2, end).trim().toLowerCase();
      // pop up to the matching open tag
      let node = cur;
      while (node && node.tag !== name) node = node.parent;
      if (node && node.parent) cur = node.parent;
      i = end + 1;
      continue;
    }
    if (html[i] === '<') { // opening tag
      const end = html.indexOf('>', i);
      if (end === -1) { pushText(html.slice(i)); break; }
      let raw = html.slice(i + 1, end);
      const selfClose = raw.endsWith('/');
      if (selfClose) raw = raw.slice(0, -1);
      const sp = raw.search(/\s/);
      const tag = (sp === -1 ? raw : raw.slice(0, sp)).toLowerCase();
      const attrStr = sp === -1 ? '' : raw.slice(sp + 1);
      const attrs = {};
      const attrRe = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*(?:=\s*("([^"]*)"|'([^']*)'|(\S+)))?/g;
      let am;
      while ((am = attrRe.exec(attrStr)) !== null) {
        const key = am[1].toLowerCase();
        const val = am[3] ?? am[4] ?? am[5] ?? '';
        attrs[key] = val;
      }
      const node = { tag, attrs, children: [], parent: cur };
      cur.children.push(node);
      i = end + 1;
      if (RAWTEXT.has(tag) && !selfClose) {
        const closeRe = new RegExp(`</${tag}\\s*>`, 'i');
        const rest = html.slice(i);
        const cm = rest.search(closeRe);
        const text = cm === -1 ? rest : rest.slice(0, cm);
        if (text) node.children.push({ tag: '#text', text, parent: node });
        i = cm === -1 ? n : i + cm + rest.slice(cm).indexOf('>') + 1;
        continue;
      }
      if (!VOID.has(tag) && !selfClose) cur = node;
      continue;
    }
    // text run
    const next = html.indexOf('<', i);
    const text = html.slice(i, next === -1 ? n : next);
    pushText(decodeEntities(text));
    i = next === -1 ? n : next;
  }
  return root;
}

function decodeEntities(s) {
  return s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ');
}

// ---------------------------------------------------------------------------
// Element wrapper exposing the DOM surface the extractors use.
// ---------------------------------------------------------------------------
function walk(node, fn) {
  fn(node);
  for (const c of node.children) if (c.tag !== '#text') walk(c, fn);
}

function textOf(node) {
  let out = '';
  for (const c of node.children) {
    if (c.tag === '#text') out += c.text;
    else if (!RAWTEXT.has(c.tag)) out += textOf(c);
  }
  return out;
}

// innerText-ish: textContent but with block elements introducing newlines and
// collapsed runs of whitespace — enough for the regex extractors and the
// document-order "first match wins" guarantee the KF fixture depends on.
const BLOCK = new Set(['div', 'p', 'li', 'ul', 'ol', 'section', 'h1', 'h2', 'h3',
  'header', 'footer', 'main', 'article', 'br', 'tr', 'table']);
function innerTextOf(node) {
  let out = '';
  function rec(n) {
    for (const c of n.children) {
      if (c.tag === '#text') { out += c.text; continue; }
      if (RAWTEXT.has(c.tag)) continue;
      const block = BLOCK.has(c.tag);
      if (block) out += '\n';
      rec(c);
      if (block) out += '\n';
    }
  }
  rec(node);
  // collapse whitespace within lines, keep line breaks
  return out.split('\n').map(l => l.replace(/\s+/g, ' ').trim()).filter(Boolean).join('\n');
}

function rawHTMLOf(node) {
  let out = '';
  for (const c of node.children) {
    if (c.tag === '#text') { out += c.text; continue; }
    const attrs = Object.entries(c.attrs).map(([k, v]) => ` ${k}="${v}"`).join('');
    if (VOID.has(c.tag)) { out += `<${c.tag}${attrs}>`; continue; }
    out += `<${c.tag}${attrs}>${rawHTMLOf(c)}</${c.tag}>`;
  }
  return out;
}

function makeElement(node) {
  if (!node) return null;
  if (node.__el) return node.__el;
  const attrs = node.attrs || {};
  const el = {
    tagName: node.tag.toUpperCase(),
    get textContent() { return textOf(node).replace(/\s+/g, ' ').trim(); },
    get innerText() { return innerTextOf(node); },
    get innerHTML() { return rawHTMLOf(node); },
    get href() { return attrs.href || ''; },
    get src() { return attrs.src || ''; },
    get id() { return attrs.id || ''; },
    dataset: Object.fromEntries(
      Object.entries(attrs).filter(([k]) => k.startsWith('data-'))
        .map(([k, v]) => [k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase()), v])
    ),
    getAttribute(name) { return attrs[name.toLowerCase()] ?? null; },
    querySelector(sel) { return querySelector(node, sel); },
    querySelectorAll(sel) { return querySelectorAll(node, sel); },
    addEventListener() {}, click() {}, remove() {}, appendChild() {},
  };
  node.__el = el;
  return el;
}

// ---------------------------------------------------------------------------
// Minimal CSS selector matching (the exact shapes the extractors pass).
// Supports: comma lists, descendant combinators, tag, .class, [attr],
// [attr*="v"] / [attr*="v" i], [attr^=], [role="main"].
// ---------------------------------------------------------------------------
function parseSimple(part) {
  // returns predicate(node)
  const conds = [];
  let s = part.trim();
  // attribute selectors [a*="b" i] etc.
  s = s.replace(/\[([^\]]+)\]/g, (_, body) => {
    const m = body.match(/^\s*([-a-zA-Z0-9_:]+)\s*(?:([*^$~|]?=)\s*(?:"([^"]*)"|'([^']*)'|([^\s\]]+)))?\s*(i)?\s*$/);
    if (m) {
      const attr = m[1].toLowerCase();
      const op = m[2];
      const val = (m[3] ?? m[4] ?? m[5] ?? '');
      const ci = !!m[6];
      conds.push((node) => {
        const av = node.attrs[attr];
        if (av === undefined) return false;
        if (!op) return true;
        const a = ci ? av.toLowerCase() : av;
        const b = ci ? val.toLowerCase() : val;
        if (op === '=') return a === b;
        if (op === '*=') return a.includes(b);
        if (op === '^=') return a.startsWith(b);
        if (op === '$=') return a.endsWith(b);
        return false;
      });
    }
    return '';
  });
  // classes
  s.replace(/\.([-a-zA-Z0-9_]+)/g, (_, cls) => {
    conds.push((node) => (node.attrs.class || '').split(/\s+/).includes(cls));
    return '';
  });
  s = s.replace(/\.([-a-zA-Z0-9_]+)/g, '');
  // leading tag name
  const tagMatch = s.trim().match(/^([a-zA-Z][-a-zA-Z0-9]*)/);
  if (tagMatch) {
    const tag = tagMatch[1].toLowerCase();
    conds.push((node) => node.tag === tag);
  }
  return (node) => node.tag !== '#text' && conds.every(c => c(node));
}

function compileSelector(selector) {
  // comma list -> array of compound selectors (each a list of descendant simples)
  return selector.split(',').map(s => s.trim()).filter(Boolean).map(compound => {
    const simples = compound.split(/\s+/).filter(Boolean).map(parseSimple);
    // match node against the LAST simple, then verify an ancestor chain matches the rest
    return (node) => {
      if (!simples[simples.length - 1](node)) return false;
      let idx = simples.length - 2;
      let anc = node.parent;
      while (idx >= 0 && anc) {
        if (simples[idx](anc)) idx--;
        anc = anc.parent;
      }
      return idx < 0;
    };
  });
}

function querySelectorAll(root, selector) {
  const matchers = compileSelector(selector);
  const out = [];
  walk(root, (node) => {
    if (node === root) return;
    if (matchers.some(m => m(node))) out.push(makeElement(node));
  });
  return out;
}

function querySelector(root, selector) {
  const all = querySelectorAll(root, selector);
  return all.length ? all[0] : null;
}

// ---------------------------------------------------------------------------
// Public: load an extractor closure out of content.js bound to a fixture DOM.
// hostname drives the script's detectSite() => currentSite.
// ---------------------------------------------------------------------------
export function loadExtractors(html, { hostname, pathname = '/properties/1', href } = {}) {
  const tree = parseHTML(html);
  // locate <body>, <main>, <title>
  let bodyNode = null, titleText = '';
  walk(tree, (node) => {
    if (node.tag === 'body' && !bodyNode) bodyNode = node;
    if (node.tag === 'title' && !titleText) titleText = textOf(node).trim();
  });
  if (!bodyNode) bodyNode = tree;

  const exported = {};
  const documentShim = {
    title: titleText,
    get body() { return makeElement(bodyNode); },
    getElementById(id) {
      let found = null;
      walk(tree, (node) => { if (!found && node.attrs && node.attrs.id === id) found = node; });
      return makeElement(found);
    },
    querySelector(sel) { return querySelector(tree, sel); },
    querySelectorAll(sel) { return querySelectorAll(tree, sel); },
    // Mutable throwaway element for the script's own DOM-building (injectError /
    // popup render). Must allow property assignment (id/textContent/innerHTML/style),
    // so it is a plain object, NOT a getter-backed fixture element.
    createElement: () => ({
      id: '', className: '', textContent: '', innerHTML: '', innerText: '',
      href: '', src: '', style: {}, dataset: {},
      setAttribute() {}, getAttribute: () => null,
      appendChild() {}, append() {}, remove() {},
      addEventListener() {}, click() {},
      querySelector: () => null, querySelectorAll: () => [],
      classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    }),
    addEventListener() {},
  };
  const windowShim = {
    location: {
      hostname,
      pathname,
      href: href || `https://${hostname}${pathname}`,
    },
    addEventListener() {},
    __rentFairValueLoaded: undefined,
  };
  windowShim.window = windowShim;

  const sandbox = {
    window: windowShim,
    document: documentShim,
    history: { pushState() {}, replaceState() {} },
    location: windowShim.location,
    setInterval: () => 0, clearInterval: () => {},
    setTimeout: () => 0, clearTimeout: () => {},
    MutationObserver: class { observe() {} disconnect() {} },
    fetch: async () => ({ ok: false, status: 599, json: async () => ({}) }),
    Tesseract: undefined,
    chrome: { runtime: { sendMessage: () => {}, getURL: () => '', lastError: null } },
    console: { log() {}, warn() {}, error() {} },
    __export: exported,
  };

  const EXPORT_SHIM = '\n;try{' +
    '__export.extractPropertyDataRightmove=extractPropertyDataRightmove;' +
    '__export.extractPropertyDataKnightFrank=extractPropertyDataKnightFrank;' +
    '__export.extractPropertyDataFoxtons=extractPropertyDataFoxtons;' +
    '__export.extractPropertyDataChestertons=extractPropertyDataChestertons;' +
    '__export.extractSqftFromPage=extractSqftFromPage;' +
    '__export.extractPostcode=extractPostcode;' +
    '__export.parsePrice=parsePrice;' +
    '__export.getFloorplanUrl=getFloorplanUrl;' +
    '__export.extractPropertyType=extractPropertyType;' +
    '__export.extractLetType=extractLetType;' +
    '__export.currentSite=currentSite;' +
    '}catch(e){__export.err=e;}\n';
  const closeIdx = SRC.lastIndexOf('})();');
  if (closeIdx === -1) throw new Error('could not find IIFE close `})();` in content.js');
  const instrumented = SRC.slice(0, closeIdx) + EXPORT_SHIM + SRC.slice(closeIdx);
  const names = Object.keys(sandbox);
  const vals = names.map(n => sandbox[n]);
  // eslint-disable-next-line no-new-func
  const fn = new Function(...names, `'use strict';\n${instrumented}`);
  fn(...vals);
  if (exported.err) throw exported.err;
  return exported;
}

export { SRC };
