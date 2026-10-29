import { EventEmitter } from 'node:events';
import { describe, expect, it, vi } from 'vitest';
import { reviewBackendProxy } from '../../backendProxy';

function configuredProxy() {
  const events = new EventEmitter();
  const options = reviewBackendProxy();
  const configure = options.configure!;
  configure(events as Parameters<typeof configure>[0], options);
  return events;
}

function responseState(overrides = {}) {
  return {
    headersSent: false,
    writableEnded: false,
    destroyed: false,
    writeHead: vi.fn(),
    end: vi.fn(),
    ...overrides,
  };
}

describe('Backend proxy transport errors', () => {
  it('returns a stable unavailable response without disclosing the transport exception', () => {
    const proxy = configuredProxy();
    const response = responseState();
    proxy.emit('error', new Error('ECONNREFUSED private transport details'), {}, response);
    expect(response.writeHead).toHaveBeenCalledWith(502, {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
    });
    expect(response.end).toHaveBeenCalledWith(
      JSON.stringify({ error: { code: 'BACKEND_UNREACHABLE' } }),
    );
  });

  it.each(['headersSent', 'writableEnded', 'destroyed'])(
    'does not overwrite a response when %s is true',
    (state) => {
      const response = responseState({ [state]: true });
      configuredProxy().emit('error', new Error('connection closed'), {}, response);
      expect(response.writeHead).not.toHaveBeenCalled();
      expect(response.end).not.toHaveBeenCalled();
    },
  );

  it('leaves WebSocket failures to the existing proxy handler', () => {
    const socket = { end: vi.fn() };
    configuredProxy().emit('error', new Error('connection closed'), {}, socket);
    expect(socket.end).not.toHaveBeenCalled();
  });
});
