import { useApi } from '../hooks/useApi';
import { Gauge } from './Gauge';
import { Panel } from './Panel';

interface Model {
  id: string;
  type: string;
  context_length: number;
  status: string;
  memory_mb: number;
  pid: number | null;
}

interface ModelsData {
  models: Model[];
  engine_status: string;
  total_models: number;
}

interface DashboardData {
  system: {
    ram_total_gb: number;
  };
}

function formatCtx(n: number): string {
  if (n >= 1024) return `${Math.round(n / 1024)}K`;
  return `${n}`;
}

function formatMem(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)}GB`;
  return `${Math.round(mb)}MB`;
}

export function ModelPanel() {
  const { data } = useApi<ModelsData>('/api/models', 5000);
  const { data: dashData } = useApi<DashboardData>('/api/dashboard', 10000);

  const totalRamGb = dashData?.system?.ram_total_gb || 64;
  const statusColor = data?.engine_status === 'ok' ? 'var(--green)' : 'var(--red)';
  const statusLabel = data?.engine_status === 'ok' ? 'ONLINE' : 'OFFLINE';

  return (
    <Panel title="Models Loaded" icon="⬡" accentColor={statusColor}>
      {!data ? (
        <div style={{ color: 'var(--text-dim)' }}>Connecting...</div>
      ) : data.models.length === 0 ? (
        <div style={{ color: 'var(--text-dim)' }}>
          Bodega {statusLabel.toLowerCase()}. No models loaded.
        </div>
      ) : (
        <div>
          {data.models.map((m) => (
            <div key={m.id} style={{
              padding: '8px 0',
              borderBottom: '1px solid var(--border)',
              animation: 'fade-in 0.3s ease',
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '4px',
              }}>
                <span style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '12px' }}>
                  {m.id}
                </span>
                <span style={{
                  fontSize: '9px',
                  padding: '1px 6px',
                  borderRadius: '2px',
                  background: m.type === 'multimodal' ? 'rgba(170,102,255,0.15)' : 'rgba(0,212,255,0.15)',
                  color: m.type === 'multimodal' ? 'var(--purple)' : 'var(--cyan)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}>
                  {m.type}
                </span>
              </div>
              <div style={{
                display: 'flex',
                gap: '12px',
                fontSize: '10px',
                color: 'var(--text-dim)',
              }}>
                <span>{formatCtx(m.context_length)} ctx</span>
                <span>{formatMem(m.memory_mb)}</span>
                {m.pid && <span>PID {m.pid}</span>}
              </div>
              {m.memory_mb > 0 && (
                <div style={{ marginTop: '4px' }}>
                  <Gauge
                    label=""
                    value={Math.min((m.memory_mb / 1024 / totalRamGb) * 100, 100)}
                    color={m.type === 'multimodal' ? 'var(--purple)' : 'var(--cyan)'}
                    suffix="GB"
                  />
                </div>
              )}
            </div>
          ))}
          <div style={{
            marginTop: '8px',
            fontSize: '10px',
            color: 'var(--text-dim)',
            display: 'flex',
            justifyContent: 'space-between',
          }}>
            <span>{data.total_models} model{data.total_models !== 1 ? 's' : ''}</span>
            <span style={{ color: statusColor }}>● {statusLabel}</span>
          </div>
        </div>
      )}
    </Panel>
  );
}
