/* Shared harness: ANGLE global-struct emitted-name collision probe.
 * Targets WebKit's ANGLE roll (Safari macOS/iOS) and upstream ANGLE (Chrome).
 *
 * WebKit RenameAndDeclareStruct():
 *   global struct, len >  1012  ->  emits  "struct" + sha1hex(fullname) + "_0"
 *   global struct, len <= 1012  ->  emits  truncate(name,1012) + "_0"
 * A global struct literally named "struct"+sha1hex(L) is 46 chars, so it takes the
 * UNHASHED path and emits the *same* 48-char name as long name L. Collision.
 *
 * Upstream ANGLE hashes every global struct longer than 40 chars, closing this.
 */
const HEX = b => [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, '0')).join('');
async function sha1hex(s) { return HEX(await crypto.subtle.digest('SHA-1', new TextEncoder().encode(s))); }
const rndHex = n => HEX(crypto.getRandomValues(new Uint8Array(n)));

function glctx() {
  const c = document.createElement('canvas');
  c.width = c.height = 1;
  return c.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false, depth: false, stencil: false });
}

/* Build the two struct source names for a given test case. */
async function names(kind, seedHex) {
  // L must exceed 1012 chars to take WebKit's hashed path.
  const L = 'Q' + seedHex + 'q'.repeat(1013 - 1 - seedHex.length);
  const good = 'struct' + await sha1hex(L);                 // collides with L's emitted name
  const bad  = 'struct' + await sha1hex(L + 'x');           // same length, WRONG hash -> no collision
  switch (kind) {
    case 'T1':  return { inBlock: L,   big: good, note: 'second-order collision (L hashed, S unhashed)' };
    case 'T1r': return { inBlock: good, big: L,   note: 'same, reversed declaration order' };
    case 'T2':  return { inBlock: 'A'.repeat(1012) + 'BBBBBBBB', big: 'A'.repeat(1012) + 'CCCCCCCC',
                         note: 'original Chrome vector: both >1012, both hashed on distinct full names' };
    case 'T3':  return { inBlock: L,   big: bad,  note: 'LENGTH-MATCHED CONTROL: wrong hash, no collision' };
    case 'T2c': {
      // Length-matched control for T2: both names are 1020 chars, but they differ at
      // index 1000, INSIDE the 1012-char truncation window. Truncation therefore cannot
      // collapse them. Same lengths, same code path, collision removed -- this is what
      // isolates the collision as the cause rather than "long identifiers break things".
      const a = 'A'.repeat(1000) + 'X' + 'A'.repeat(11) + 'BBBBBBBB';
      const b = 'A'.repeat(1000) + 'Y' + 'A'.repeat(11) + 'CCCCCCCC';
      return { inBlock: a, big: b, note: 'LENGTH-MATCHED CONTROL for T2: differ at idx 1000 (<1012), no collision' };
    }
    case 'T4':  return { inBlock: 'ShortA01', big: 'ShortB01', note: 'baseline control: short names' };
    default: throw Error('unknown kind ' + kind);
  }
}

function fragSrc(inBlock, big, n) {
  return `#version 300 es
precision highp float;
struct ${inBlock} { vec4 v[1]; };
struct ${big} { vec4 v[${n}]; };
layout(std140) uniform Blk { ${inBlock} s; vec4 tail; } blk;
uniform ${big} keepalive;
uniform int mode;
out vec4 color;
void main(){ color = mode==0 ? blk.s.v[0] : (mode==1 ? blk.tail : keepalive.v[0]); }`;
}

/* crossStage: also declare both structs in the vertex shader, which is the
 * cross-stage linking path the original report described but never exercised. */
function vertSrc(inBlock, big, n, crossStage) {
  if (!crossStage) return `#version 300 es
void main(){ gl_Position = vec4(0,0,0,1); gl_PointSize = 1.0; }`;
  return `#version 300 es
struct ${inBlock} { vec4 v[1]; };
struct ${big} { vec4 v[${n}]; };
layout(std140) uniform Blk { ${inBlock} s; vec4 tail; } blk;
void main(){ gl_Position = vec4(blk.tail.x*0.0, 0, 0, 1); gl_PointSize = 1.0; }`;
}

function buildProgram(gl, inBlock, big, n, crossStage) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error('COMPILE: ' + gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vertSrc(inBlock, big, n, crossStage)));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fragSrc(inBlock, big, n)));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error('LINK: ' + gl.getProgramInfoLog(p));
  gl.useProgram(p);

  const bi = gl.getUniformBlockIndex(p, 'Blk');
  gl.uniformBlockBinding(p, bi, 0);
  // Reflection: what ANGLE *reports* to the API, computed from the AST.
  const blockSize = gl.getActiveUniformBlockParameter(p, bi, gl.UNIFORM_BLOCK_DATA_SIZE);
  const idx = gl.getActiveUniformBlockParameter(p, bi, gl.UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES);
  const offsets = {};
  for (const i of idx) {
    const info = gl.getActiveUniform(p, i);
    offsets[info.name] = gl.getActiveUniforms(p, [i], gl.UNIFORM_OFFSET)[0];
  }
  const kl = gl.getUniformLocation(p, 'keepalive.v[0]');
  if (kl) gl.uniform4f(kl, 1, 1, 1, 1);
  const mode = gl.getUniformLocation(p, 'mode');
  const px = new Uint8Array(4);
  return {
    n, blockSize, offsets,
    read(m) {
      gl.useProgram(p); gl.uniform1i(mode, m);
      gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.POINTS, 0, 1);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      return [...px];
    }
  };
}

/* 4096-byte sentinel: slot k encodes its own index in red, 0xA5 in blue.
 * Reading tail and getting red==k proves tail was fetched from offset 16*k. */
function sentinel(gl, bytes = 4096) {
  const f = new Float32Array(bytes / 4);
  for (let k = 0; k * 4 < f.length; k++) {
    f[k * 4 + 0] = (k & 255) / 255;
    f[k * 4 + 1] = ((k >> 8) & 255) / 255;
    f[k * 4 + 2] = 0xA5 / 255;
    f[k * 4 + 3] = 1;
  }
  const b = gl.createBuffer();
  gl.bindBuffer(gl.UNIFORM_BUFFER, b);
  gl.bufferData(gl.UNIFORM_BUFFER, f, gl.STATIC_DRAW);
  return b;
}

/* 32-byte buffer: both vec4 slots carry the same id marker. */
function marked(gl, r, g, b_, a, bytes = 32) {
  const f = new Float32Array(bytes / 4);
  for (let i = 0; i < 8; i += 4) { f[i] = r / 255; f[i + 1] = g / 255; f[i + 2] = b_ / 255; f[i + 3] = a / 255; }
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.UNIFORM_BUFFER, buf);
  gl.bufferData(gl.UNIFORM_BUFFER, f, gl.STATIC_DRAW);
  return buf;
}

const bindRange = (gl, b, size = 32) => gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, b, 0, size);
const yieldFrame = () => new Promise(r => requestAnimationFrame(() => r()));

function env(gl) {
  const e = {};
  try { const x = gl.getExtension('WEBGL_debug_renderer_info');
        e.renderer = x ? gl.getParameter(x.UNMASKED_RENDERER_WEBGL) : 'masked'; } catch { e.renderer = 'masked'; }
  e.vendor = gl.getParameter(gl.VENDOR);
  e.version = gl.getParameter(gl.VERSION);
  e.maxFragVec = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS);
  e.maxVertVec = gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS);
  e.maxUBOSize = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE);
  e.uboAlign = gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT);
  e.ua = navigator.userAgent;
  return e;
}
