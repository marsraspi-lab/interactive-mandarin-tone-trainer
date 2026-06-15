#!/usr/bin/env node
/**
 * Minimal static file server for local development.
 * Serves src/ at localhost:$PORT (default 3000).
 *
 * Usage: node scripts/dev-server.js [port]
 */

import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { extname, join, normalize } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', 'src');
const PORT = parseInt(process.argv[2] || '3000', 10);

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.mjs':  'application/javascript',
  '.css':  'text/css',
  '.json': 'application/json',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.mp3':  'audio/mpeg',
  '.wav':  'audio/wav',
  '.svg':  'image/svg+xml',
  '.ico':  'image/x-icon',
};

const server = createServer(async (req, res) => {
  // Sanitize path (prevent traversal)
  let urlPath = req.url.split('?')[0];
  urlPath = normalize(urlPath).replace(/^(\.\.[/\\])+/, '');
  if (urlPath === '/' || urlPath === '') urlPath = '/index.html';

  const filePath = join(ROOT, urlPath);

  try {
    const data = await readFile(filePath);
    const ext = extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`Dev server → http://localhost:${PORT}`);
  console.log(`Admin     → http://localhost:${PORT}/admin.html`);
  console.log(`Serving   → ${ROOT}`);
});
