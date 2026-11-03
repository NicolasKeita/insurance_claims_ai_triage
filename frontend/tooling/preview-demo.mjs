import { createServer } from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { frontendRoot } from './prepare.mjs';

const root = await realpath(path.join(frontendRoot, 'out'));
const manifest = JSON.parse(await readFile(path.join(root, 'insurance-demo-build.json'), 'utf8'));
if (manifest.runtime !== 'static') throw new Error('Build the static demo before previewing it');
const base = `${manifest.basePath || ''}/`;
const portIndex = process.argv.indexOf('--port');
const port = Number(portIndex < 0 ? 4173 : process.argv[portIndex + 1]);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid preview port');
const types = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
};
createServer(async (request, response) => {
  if (!['GET', 'HEAD'].includes(request.method)) {
    response.writeHead(405, { Allow: 'GET, HEAD' }).end();
    return;
  }
  try {
    const url = new URL(request.url, 'http://localhost');
    if (manifest.basePath && url.pathname === manifest.basePath) {
      response.writeHead(308, { Location: `${base}${url.search}` }).end();
      return;
    }
    if (!url.pathname.startsWith(base)) throw new Error('Outside the demo');
    const relative = decodeURIComponent(url.pathname.slice(base.length));
    const candidate = path.resolve(root, relative || 'index.html');
    if (candidate !== root && !candidate.startsWith(root + path.sep))
      throw new Error('Outside the demo');
    const candidateStat = await stat(candidate);
    const target = await realpath(
      candidateStat.isDirectory() ? path.join(candidate, 'index.html') : candidate,
    );
    if (!target.startsWith(root + path.sep)) throw new Error('Outside the demo');
    const content = await readFile(target);
    response.writeHead(200, {
      'Content-Type': types[path.extname(target)] || 'application/octet-stream',
      'Content-Length': content.length,
      'Cache-Control': 'no-cache',
    });
    response.end(request.method === 'HEAD' ? undefined : content);
  } catch {
    response.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  process.stdout.write(`Insurance AI preview: http://127.0.0.1:${port}${base}\n`);
});
