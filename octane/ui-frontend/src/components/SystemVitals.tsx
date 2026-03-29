import { useApi } from '../hooks/useApi';
import { Gauge } from './Gauge';
import { Panel } from './Panel';

interface SystemData {
  cpu: { percent: number; cores: number };
  ram: { total_gb: number; used_gb: number; percent: number };
  disk: { total_gb: number; used_gb: number; percent: number };
  gpu: { available: boolean; chip: string; unified_memory_gb: number; memory_pressure_percent: number };
}

export function SystemVitals() {
  const { data } = useApi<SystemData>('/api/system', 3000);

  if (!data) {
    return (
      <Panel title="System Vitals" icon="◈">
        <div style={{ color: 'var(--text-dim)' }}>Connecting...</div>
      </Panel>
    );
  }

  return (
    <Panel title="System Vitals" icon="◈">
      <Gauge label="CPU" value={data.cpu.percent} />
      <Gauge label="RAM" value={data.ram.percent} color="var(--cyan)" />
      <Gauge label="GPU MEM" value={data.gpu.memory_pressure_percent || 0} color="var(--purple)" />
      <Gauge label="SSD" value={data.disk.percent} color="var(--blue)" />
      <div style={{
        marginTop: '8px',
        fontSize: '10px',
        color: 'var(--text-dim)',
        borderTop: '1px solid var(--border)',
        paddingTop: '8px',
      }}>
        {data.gpu.chip || 'Apple Silicon'} · {data.ram.total_gb}GB unified
      </div>
    </Panel>
  );
}
