import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface CorrelationData {
  tickers: string[];
  matrix: Record<string, Record<string, number>>;
  message?: string;
}

function corrColor(val: number): string {
  if (val >= 0.8) return '#00ff88';
  if (val >= 0.5) return '#00d4ff';
  if (val >= 0.2) return '#44aaff';
  if (val >= -0.2) return '#555566';
  if (val >= -0.5) return '#ffaa00';
  return '#ff4444';
}

export function CorrelationHeatmap() {
  const { data, loading } = useApi<CorrelationData>('/api/portfolio/correlation', 60000);

  if (loading) return <Panel title="Correlation" icon="▦" accentColor="var(--amber)">Loading…</Panel>;
  if (!data?.tickers?.length || !Object.keys(data.matrix).length) {
    return (
      <Panel title="Correlation" icon="▦" accentColor="var(--amber)">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          {data?.message || 'Need ≥2 tickers for correlation matrix.'}
        </div>
      </Panel>
    );
  }

  const { tickers, matrix } = data;
  const cellSize = Math.min(40, 240 / tickers.length);

  return (
    <Panel title="Correlation" icon="▦" accentColor="var(--amber)">
      <div style={{ overflowX: 'auto', padding: '8px 0' }}>
        <table style={{
          borderCollapse: 'collapse',
          margin: '0 auto',
          fontSize: '10px',
          fontFamily: 'var(--font-mono)',
        }}>
          <thead>
            <tr>
              <th style={{ padding: '4px 6px' }} />
              {tickers.map(t => (
                <th key={t} style={{
                  padding: '4px 6px',
                  color: 'var(--text-secondary)',
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                }}>
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map(t1 => (
              <tr key={t1}>
                <td style={{
                  padding: '4px 6px',
                  color: 'var(--text-secondary)',
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                }}>
                  {t1}
                </td>
                {tickers.map(t2 => {
                  const val = matrix[t1]?.[t2] ?? 0;
                  const isDiag = t1 === t2;
                  return (
                    <td
                      key={t2}
                      title={`${t1}/${t2}: ${val.toFixed(4)}`}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        textAlign: 'center',
                        background: isDiag ? 'var(--bg-hover)' : corrColor(val),
                        color: isDiag ? 'var(--text-dim)' : '#000',
                        fontWeight: 600,
                        opacity: isDiag ? 0.5 : Math.max(0.3, Math.abs(val)),
                        borderRadius: '2px',
                        padding: '2px',
                        cursor: 'default',
                      }}
                    >
                      {isDiag ? '—' : val.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '12px',
        marginTop: '8px',
        fontSize: '9px',
        color: 'var(--text-dim)',
      }}>
        <span><span style={{ color: '#ff4444' }}>■</span> Neg</span>
        <span><span style={{ color: '#555566' }}>■</span> Low</span>
        <span><span style={{ color: '#44aaff' }}>■</span> Mid</span>
        <span><span style={{ color: '#00ff88' }}>■</span> High</span>
      </div>
    </Panel>
  );
}
