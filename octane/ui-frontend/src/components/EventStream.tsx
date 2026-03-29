import { useEvents } from '../context/EventsContext';
import type { SynapseEvent } from '../hooks/useWebSocket';
import { Panel } from './Panel';

const EVENT_COLORS: Record<string, string> = {
  ingress: 'var(--green)',
  egress: 'var(--green)',
  agent_start: 'var(--cyan)',
  agent_complete: 'var(--blue)',
  decomposition: 'var(--amber)',
  web_search_round: 'var(--purple)',
  web_synthesis: 'var(--purple)',
  guard: 'var(--amber)',
  preflight: 'var(--text-dim)',
  dispatch: 'var(--cyan)',
  memory_read: 'var(--text-dim)',
  memory_write: 'var(--text-dim)',
};

const EVENT_ICONS: Record<string, string> = {
  ingress: '▸',
  egress: '◂',
  agent_start: '●',
  agent_complete: '✓',
  decomposition: '◆',
  web_search_round: '◉',
  web_synthesis: '◈',
  guard: '▲',
  preflight: '○',
  dispatch: '→',
};

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '--:--:--';
  }
}

export function EventStream() {
  const { events, connected } = useEvents();

  const recent = events.slice(-30).reverse();

  return (
    <Panel
      title="Live Events"
      icon="◎"
      accentColor={connected ? 'var(--green)' : 'var(--red)'}
    >
      {recent.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: '11px' }}>
          {connected ? 'Listening for events...' : 'Connecting...'}
        </div>
      ) : (
        <div style={{ fontSize: '11px' }}>
          {recent.map((ev: SynapseEvent) => (
            <div key={ev.event_id} style={{
              padding: '3px 0',
              borderBottom: '1px solid var(--bg-secondary)',
              display: 'flex',
              gap: '8px',
              alignItems: 'baseline',
              animation: 'fade-in 0.2s ease',
            }}>
              <span style={{ color: 'var(--text-dim)', fontSize: '10px', flexShrink: 0 }}>
                {formatTime(ev.timestamp)}
              </span>
              <span style={{
                color: EVENT_COLORS[ev.event_type] || 'var(--text-secondary)',
                flexShrink: 0,
                width: '12px',
                textAlign: 'center',
              }}>
                {EVENT_ICONS[ev.event_type] || '·'}
              </span>
              <span style={{
                color: EVENT_COLORS[ev.event_type] || 'var(--text-secondary)',
                fontSize: '10px',
                flexShrink: 0,
                minWidth: '100px',
              }}>
                {ev.event_type}
              </span>
              <span style={{ color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {ev.source}
                {ev.duration_ms > 0 && (
                  <span style={{ color: 'var(--text-dim)', marginLeft: '6px' }}>
                    {ev.duration_ms.toFixed(0)}ms
                  </span>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
