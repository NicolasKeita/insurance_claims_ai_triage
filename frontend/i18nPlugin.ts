import { spawnSync } from 'node:child_process';
import path from 'node:path';
import type { Plugin } from 'vite';

const moduleId = 'virtual:review-locales';
const resolvedId = `\0${moduleId}`;

/** TOML is parsed by tomllib, never by a partial JavaScript parser or eval. */
export function reviewI18n(): Plugin {
  let frontendRoot = '';
  let cataloguesRoot = '';
  function compile(): unknown {
    const projectRoot = path.resolve(frontendRoot, '..');
    const python = path.join(
      projectRoot,
      '.venv',
      process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
    );
    const result = spawnSync(
      python,
      [path.join(frontendRoot, 'tooling/compile_locales.py'), cataloguesRoot],
      {
        cwd: projectRoot,
        encoding: 'utf8',
        windowsHide: true,
        shell: false,
        env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
      },
    );
    if (result.error || result.status !== 0) {
      throw new Error(
        result.stderr?.trim() ||
          `Cannot parse translation TOML. Use the project's Python 3.12+ .venv. ${result.error?.message || ''}`,
      );
    }
    return JSON.parse(result.stdout);
  }
  return {
    name: 'claims-review-i18n',
    configResolved(config) {
      frontendRoot = path.resolve(config.root);
      cataloguesRoot = path.join(frontendRoot, 'locales');
    },
    buildStart() {
      compile();
    },
    resolveId(id) {
      return id === moduleId ? resolvedId : undefined;
    },
    load(id) {
      if (id !== resolvedId) return;
      this.addWatchFile(path.join(cataloguesRoot, 'fr.toml'));
      this.addWatchFile(path.join(cataloguesRoot, 'en.toml'));
      return `export default ${JSON.stringify(compile())};`;
    },
    configureServer(server) {
      server.watcher.add(cataloguesRoot);
    },
    handleHotUpdate({ file, server }) {
      if (path.resolve(path.dirname(file)) !== cataloguesRoot || !file.endsWith('.toml')) return;
      compile();
      const module = server.moduleGraph.getModuleById(resolvedId);
      if (module) server.moduleGraph.invalidateModule(module);
      server.ws.send({ type: 'full-reload', path: '*' });
      return [];
    },
  };
}
