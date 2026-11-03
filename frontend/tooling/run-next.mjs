import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { frontendRoot, prepareFrontend } from './prepare.mjs';

const [command, ...provided] = process.argv.slice(2);
if (!['dev', 'build', 'start'].includes(command)) throw new Error('Expected dev, build or start');
const demo = provided.includes('--demo');
const args = provided.filter((arg) => arg !== '--demo');
const runtime = demo ? 'static' : process.env.CLAIMS_REVIEW_RUNTIME || 'api';
const env = { ...process.env, CLAIMS_REVIEW_RUNTIME: runtime, NEXT_TELEMETRY_DISABLED: '1' };
prepareFrontend({ runtime });
if (command !== 'start') {
  const panda = spawnSync(
    process.execPath,
    [path.join(frontendRoot, 'node_modules/@pandacss/dev/bin.js'), 'codegen'],
    {
      cwd: frontendRoot,
      env,
      stdio: 'inherit',
      windowsHide: true,
    },
  );
  if (panda.error) throw panda.error;
  if (panda.status !== 0) process.exit(panda.status || 1);
}
if (command !== 'start') args.unshift('--webpack');
if (command === 'dev' || command === 'start') {
  if (!args.includes('--hostname') && !args.includes('-H')) args.push('--hostname', '127.0.0.1');
  if (!args.includes('--port') && !args.includes('-p')) args.push('--port', '5173');
}
const child = spawn(
  process.execPath,
  [path.join(frontendRoot, 'node_modules/next/dist/bin/next'), command, ...args],
  {
    cwd: frontendRoot,
    env,
    stdio: 'inherit',
    windowsHide: true,
  },
);
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
const exitCode = await new Promise((resolve, reject) => {
  child.on('error', reject);
  child.on('exit', (code) => resolve(code ?? 1));
});
if (exitCode === 0 && command === 'build' && runtime === 'static') {
  const projectRoot = path.dirname(frontendRoot);
  const version = readFileSync(path.join(projectRoot, 'pyproject.toml'), 'utf8').match(
    /^version\s*=\s*"([^"]+)"/m,
  )?.[1];
  const revision = spawnSync('git', ['rev-parse', 'HEAD'], {
    cwd: projectRoot,
    encoding: 'utf8',
    windowsHide: true,
  });
  const status = spawnSync('git', ['status', '--porcelain'], {
    cwd: projectRoot,
    encoding: 'utf8',
    windowsHide: true,
  });
  if (!version) throw new Error('Cannot identify the demo application version');
  writeFileSync(
    path.join(frontendRoot, 'out/insurance-demo-build.json'),
    `${JSON.stringify(
      {
        format: 'insurance-demo-export-v1',
        basePath: env.CLAIMS_REVIEW_BASE_PATH || '',
        applicationVersion: version,
        // Docker builds omit Git metadata; these exports cannot pass portfolio provenance checks.
        sourceRevision: revision.status === 0 ? revision.stdout.trim() : null,
        sourceDirty: status.status !== 0 || !!status.stdout.trim(),
        runtime,
      },
      null,
      2,
    )}\n`,
  );
}
process.exit(exitCode);
