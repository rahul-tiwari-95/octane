import { useApi } from '../hooks/useApi';
import { Panel } from './Panel';

interface Trace {
  trace_id: string;
  query: string;
  timestamp: string;
  duration_ms: number;
  event_count: number;
  success: boolean;
}

interface DashboardData {
  uptime: string;
  recent_traces: Trace[];
}

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

function timeAgo(ts: string): string {
  try {
    const diff = Date.now() - new Date(ts).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  } catch {
    return '';
  }
}

export function QueryTimeline() {
  const { data } = useApi<DashboardData>('/api/dashboard', 5000);

  const traces = data?.recent_traces || [];

  return (
    <Panel title="Recent Queries" icon="▸">
      {traces.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: '11px' }}>
          No queries yet. Run <span style={{ color: 'var(--green)' }}>octane ask</span> to get started.
        </div>
      ) : (
        <div>
          {traces.map((t) => (
            <div key={t.trace_id} style={{
              padding: '6px 0',
              borderBottom: '1px solid var(--bg-secondary)',
              animation: 'fade-in 0.3s ease',
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '2px',
              }}>
                <span style={{
                  color: 'var(--text-primary)',
                  fontSize: '12px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  maxWidth: '70%',
                }}>
                  {t.query || `trace:${t.trace_id.slice(0, 8)}`}
                </span>
                <span style={{
                  fontSize: '10px',
                  color: t.success ? 'var(--green)' : 'var(--red)',
                }}>
                  {t.success ? '✓' : '✗'} {formatDuration(t.duration_ms)}
                </span>
              </div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '10px',
                color: 'var(--text-dim)',
              }}>
                <span>{t.trace_id.slice(0, 8)}</span>
                <span>{t.event_count} events</span>
                <span>{timeAgo(t.timestamp)}</span>
              </div>
              {/* Progress bar */}
              <div style={{
                marginTop: '3px',
                height: '2px',
                background: 'var(--bg-secondary)',
                borderRadius: '1px',
                overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%',
                  width: '100%',
                  background: t.success
                    ? 'linear-gradient(90deg, var(--green-dim), var(--green))'
                    : 'linear-gradient(90deg, var(--red), var(--amber))',
                  borderRadius: '1px',
                }} />
              </div>
            </div>
          ))}
          {data?.uptime && (
            <div style={{
              marginTop: '8px',
              fontSize: '10px',
              color: 'var(--text-dim)',
              textAlign: 'right',
            }}>
              UPTIME {data.uptime}
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
