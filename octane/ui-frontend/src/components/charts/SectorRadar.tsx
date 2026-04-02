import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface SectorData {
  sectors: { sector: string; value: number; weight: number }[];
  total: number;
}

export function SectorRadar() {
  const { data, loading } = useApi<SectorData>('/api/portfolio/sectors', 30000);

  if (loading) return <Panel title="Sector Exposure" icon="⬡" accentColor="var(--cyan)">Loading…</Panel>;
  if (!data?.sectors?.length) {
    return (
      <Panel title="Sector Exposure" icon="⬡" accentColor="var(--cyan)">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No sector data available.
        </div>
      </Panel>
    );
  }

  const chartData = data.sectors.map(s => ({
    subject: s.sector.length > 12 ? s.sector.slice(0, 11) + '…' : s.sector,
    fullName: s.sector,
    weight: s.weight,
    value: s.value,
  }));

  return (
    <Panel title="Sector Exposure" icon="⬡" accentColor="var(--cyan)">
      <ResponsiveContainer width="100%" height={280}>
        <RadarChart data={chartData} cx="50%" cy="50%" outerRadius="70%">
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis
            dataKey="subject"
            tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
          />
          <PolarRadiusAxis
            tick={{ fill: 'var(--text-dim)', fontSize: 9 }}
            domain={[0, 'auto']}
          />
          <Radar
            name="Weight %"
            dataKey="weight"
            stroke="#00d4ff"
            fill="#00d4ff"
            fillOpacity={0.2}
            strokeWidth={2}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-panel)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-primary)',
              fontSize: '12px',
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Weight']}
          />
        </RadarChart>
      </ResponsiveContainer>
    </Panel>
  );
}
