import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ResolvedConfig } from 'vite';
import { reviewI18n } from '../../i18nPlugin';

vi.mock('node:child_process', () => {
  const spawnSync = vi.fn();
  return { spawnSync, default: { spawnSync } };
});

const frontendRoot = path.resolve('frontend-fixture');
const projectRoot = path.dirname(frontendRoot);
const pythonResult = {
  pid: 1,
  output: [null, '{}', ''],
  stdout: '{}',
  stderr: '',
  status: 0,
  signal: null,
};

function buildCatalogues() {
  const plugin = reviewI18n();
  if (typeof plugin.configResolved !== 'function' || typeof plugin.buildStart !== 'function') {
    throw new Error('Expected configuration and build hooks');
  }
  Reflect.apply(plugin.configResolved, {}, [{ root: frontendRoot } as ResolvedConfig]);
  return Reflect.apply(plugin.buildStart, {}, []);
}

beforeEach(() => {
  vi.stubEnv('CLAIMS_I18N_PYTHON', '');
  vi.mocked(spawnSync).mockImplementation(() => pythonResult as never);
});

afterEach(() => vi.unstubAllEnvs());

describe('Translation compiler interpreter', () => {
  it('keeps the project virtual environment as the default', () => {
    buildCatalogues();
    expect(spawnSync).toHaveBeenCalledWith(
      path.join(
        projectRoot,
        '.venv',
        process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
      ),
      [path.join(frontendRoot, 'tooling/compile_locales.py'), path.join(frontendRoot, 'locales')],
      expect.objectContaining({ shell: false, encoding: 'utf8', cwd: projectRoot }),
    );
  });

  it('uses the explicit Docker interpreter as one executable without a shell', () => {
    const executable = '/opt/Python 3.12/bin/python3';
    vi.stubEnv('CLAIMS_I18N_PYTHON', executable);
    buildCatalogues();
    expect(spawnSync).toHaveBeenCalledWith(
      executable,
      [path.join(frontendRoot, 'tooling/compile_locales.py'), path.join(frontendRoot, 'locales')],
      expect.objectContaining({
        shell: false,
        env: expect.objectContaining({ PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }),
      }),
    );
  });

  it('identifies an unusable override and preserves the launch failure', () => {
    vi.stubEnv('CLAIMS_I18N_PYTHON', '/missing/python3');
    vi.mocked(spawnSync).mockImplementation(
      () => ({ ...pythonResult, status: null, error: new Error('spawn ENOENT') }) as never,
    );
    expect(buildCatalogues).toThrow(
      /Check CLAIMS_I18N_PYTHON \(\/missing\/python3\); Python 3\.12\+ is required\.[\s\S]*spawn ENOENT/,
    );
  });

  it('keeps parser diagnostics when the selected Python exits unsuccessfully', () => {
    vi.stubEnv('CLAIMS_I18N_PYTHON', '/usr/local/bin/python3');
    vi.mocked(spawnSync).mockImplementation(
      () =>
        ({
          ...pythonResult,
          status: 1,
          stderr: 'Translation catalogue error: mismatched keys',
        }) as never,
    );
    expect(buildCatalogues).toThrow(
      /CLAIMS_I18N_PYTHON[\s\S]*Translation catalogue error: mismatched keys/,
    );
  });
});
