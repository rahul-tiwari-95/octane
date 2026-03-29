import { useEffect, useRef, useState, useCallback } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import '@xterm/xterm/css/xterm.css';

/* ── Module-level session store — survives React re-renders & route changes ── */

interface TermSession {
  id: string;
  title: string;
  term: Terminal;
  fit: FitAddon;
  ws: WebSocket | null;
  container: HTMLDivElement;
  opened: boolean; // whether term.open() has been called
}

const sessionStore: Map<string, TermSession> = new Map();

const TERM_THEME = {
  background: '#0a0a0f',
  foreground: '#d4d4d4',
  cursor: '#00ff88',
  cursorAccent: '#0a0a0f',
  selectionBackground: '#00ff8833',
  black: '#0a0a0f',
  red: '#ff3366',
  green: '#00ff88',
  yellow: '#ffaa00',
  blue: '#00ccff',
  magenta: '#cc66ff',
  cyan: '#00ccff',
  white: '#d4d4d4',
  brightBlack: '#555555',
  brightRed: '#ff6699',
  brightGreen: '#33ffaa',
  brightYellow: '#ffcc33',
  brightBlue: '#33ddff',
  brightMagenta: '#dd99ff',
  brightCyan: '#33ddff',
  brightWhite: '#ffffff',
};

function makeTermSession(sessionId: string, title: string): TermSession {
  const term = new Terminal({
    cursorBlink: true,
    cursorStyle: 'block',
    fontSize: 13,
    fontFamily: "'SF Mono', 'JetBrains Mono', 'Fira Code', monospace",
    theme: TERM_THEME,
    allowProposedApi: true,
  });

  const fit = new FitAddon();
  term.loadAddon(fit);
  term.loadAddon(new WebLinksAddon());

  const container = document.createElement('div');
  container.style.cssText = 'width:100%;height:100%;';

  const session: TermSession = {
    id: sessionId,
    title,
    term,
    fit,
    ws: null,
    container,
    opened: false,
  };

  // Wire keystrokes → WebSocket (registered once per session)
  term.onData((data) => {
    if (session.ws?.readyState === WebSocket.OPEN) {
      session.ws.send(data);
    }
  });
  term.onResize(({ cols, rows }) => {
    if (session.ws?.readyState === WebSocket.OPEN) {
      session.ws.send(JSON.stringify({ type: 'resize', cols, rows }));
    }
  });

  sessionStore.set(sessionId, session);
  return session;
}

function connectSession(session: TermSession) {
  const cur = session.ws;
  if (cur && (cur.readyState === WebSocket.OPEN || cur.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(
    `${protocol}://${window.location.host}/ws/terminal?session_id=${session.id}`,
  );

  let gotSessionInfo = false;

  ws.onopen = () => {
    ws.send(
      JSON.stringify({ type: 'resize', cols: session.term.cols, rows: session.term.rows }),
    );
  };

  ws.onmessage = (event) => {
    // First message is JSON session_info — handle it, don't write to terminal
    if (!gotSessionInfo) {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.type === 'session_info') {
          gotSessionInfo = true;
          // Server may have assigned a different session_id
          if (parsed.session_id !== session.id) {
            sessionStore.delete(session.id);
            session.id = parsed.session_id;
            sessionStore.set(session.id, session);
          }
          return;
        }
      } catch {
        // not JSON — write below
      }
    }
    session.term.write(event.data);
  };

  ws.onclose = () => {
    session.ws = null;
    // Auto-reconnect after 3s if session still exists
    setTimeout(() => {
      if (sessionStore.has(session.id)) {
        connectSession(session);
      }
    }, 3000);
  };

  ws.onerror = () => {
    session.term.writeln('\x1b[31m● Connection error\x1b[0m');
  };

  session.ws = ws;
}

/* ── React component ──────────────────────────────────────────────────────── */

export function TerminalPage({ visible }: { visible?: boolean }) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [sessions, setSessions] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  // On mount: fetch existing server sessions or create a first one
  useEffect(() => {
    let cancelled = false;

    (async () => {
      // If we already have sessions in the store (navigated away and back), reuse them
      if (sessionStore.size > 0) {
        const ids = Array.from(sessionStore.keys());
        if (!cancelled) {
          setSessions(ids);
          setActiveId(ids[0]);
        }
        return;
      }

      try {
        const resp = await fetch('/api/terminal/sessions');
        const data = await resp.json();
        const alive = (data.sessions ?? []).filter((s: any) => s.alive);

        if (cancelled) return;

        if (alive.length > 0) {
          const ids: string[] = [];
          for (const s of alive) {
            if (!sessionStore.has(s.session_id)) {
              makeTermSession(s.session_id, s.title || `shell ${ids.length + 1}`);
            }
            ids.push(s.session_id);
          }
          setSessions(ids);
          setActiveId(ids[0]);
        } else {
          const resp2 = await fetch('/api/terminal/sessions', { method: 'POST' });
          const data2 = await resp2.json();
          if (cancelled) return;
          makeTermSession(data2.session_id, 'shell 1');
          setSessions([data2.session_id]);
          setActiveId(data2.session_id);
        }
      } catch {
        // Backend not running
      }
    })();

    return () => { cancelled = true; };
  }, []);

  // Attach active terminal to the DOM wrapper
  useEffect(() => {
    if (!activeId || !wrapperRef.current) return;
    const session = sessionStore.get(activeId);
    if (!session) return;

    const wrapper = wrapperRef.current;

    // Remove any previous terminal container
    while (wrapper.firstChild) {
      wrapper.removeChild(wrapper.firstChild);
    }

    // Attach this session's container
    wrapper.appendChild(session.container);

    // Open terminal into the container (only once per session)
    if (!session.opened) {
      session.term.open(session.container);
      session.opened = true;
    }

    // Re-fit after DOM attach
    requestAnimationFrame(() => session.fit.fit());

    // Connect WebSocket if not already connected
    connectSession(session);
  }, [activeId]);

  // Re-fit when this page becomes visible (after navigating back from Dashboard)
  useEffect(() => {
    if (visible && activeId) {
      const session = sessionStore.get(activeId);
      if (session) {
        requestAnimationFrame(() => session.fit.fit());
      }
    }
  }, [visible, activeId]);

  // Fit on window resize
  useEffect(() => {
    const onResize = () => {
      if (activeId) {
        sessionStore.get(activeId)?.fit.fit();
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [activeId]);

  const addSession = useCallback(async () => {
    try {
      const resp = await fetch('/api/terminal/sessions', { method: 'POST' });
      const data = await resp.json();
      const title = `shell ${sessions.length + 1}`;
      makeTermSession(data.session_id, title);
      setSessions((prev) => [...prev, data.session_id]);
      setActiveId(data.session_id);
    } catch {
      // backend error
    }
  }, [sessions.length]);

  const closeSession = useCallback(async (id: string) => {
    const session = sessionStore.get(id);
    if (session) {
      session.ws?.close();
      session.term.dispose();
      sessionStore.delete(id);
    }
    fetch(`/api/terminal/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }).catch(() => {});

    setSessions((prev) => {
      const next = prev.filter((s) => s !== id);
      if (activeId === id) {
        setActiveId(next[0] ?? null);
      }
      return next;
    });
  }, [activeId]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#0a0a0f',
    }}>
      {/* Session tab bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0 8px',
        background: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border)',
        fontSize: '10px',
        color: 'var(--text-dim)',
        flexShrink: 0,
        height: '32px',
      }}>
        {sessions.map((id) => {
          const session = sessionStore.get(id);
          const isActive = id === activeId;
          return (
            <div
              key={id}
              onClick={() => setActiveId(id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '0 12px',
                height: '100%',
                cursor: 'pointer',
                borderBottom: isActive ? '1px solid var(--green)' : '1px solid transparent',
                color: isActive ? 'var(--green)' : 'var(--text-dim)',
                background: isActive ? 'rgba(0, 255, 136, 0.05)' : 'transparent',
                transition: 'all 0.15s ease',
              }}
            >
              <span>▸</span>
              <span>{session?.title || id.slice(0, 8)}</span>
              {sessions.length > 1 && (
                <span
                  onClick={(e) => { e.stopPropagation(); closeSession(id); }}
                  style={{
                    marginLeft: '4px',
                    opacity: 0.5,
                    cursor: 'pointer',
                    fontSize: '12px',
                    lineHeight: 1,
                  }}
                  onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.opacity = '1')}
                  onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.opacity = '0.5')}
                >
                  ×
                </span>
              )}
            </div>
          );
        })}
        <div
          onClick={addSession}
          style={{
            padding: '0 10px',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            color: 'var(--text-dim)',
            fontSize: '14px',
            opacity: 0.6,
          }}
          onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.opacity = '1')}
          onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.opacity = '0.6')}
          title="New session"
        >
          +
        </div>
        <span style={{ marginLeft: 'auto', letterSpacing: '1px', padding: '0 8px' }}>
          PTY over WebSocket
        </span>
      </div>

      {/* Terminal container */}
      <div
        ref={wrapperRef}
        style={{
          flex: 1,
          padding: '4px',
          overflow: 'hidden',
        }}
      />
    </div>
  );
}
