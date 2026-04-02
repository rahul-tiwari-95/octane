import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface SourceData {
  sources: { source_type: string; count: number }[];
  total: number;
}

const COLORS = ['#00ff88', '#00d4ff', '#ffaa00', '#ff4444', '#4488ff', '#aa66ff', '#ff66aa', '#66ffaa'];

export function SourceDistribution() {
  const { data, loading } = useApi<SourceData>('/api/charts/source-distribution', 60000);

  if (loading) return <Panel title="Source Distribution" icon="◉">Loading…</Panel>;
  if (!data?.sources?.length) {
    return (
      <Panel title="Source Distribution" icon="◉">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No extractions yet.<br />
          <span style={{ fontSize: '11px' }}>Try: octane research extract &lt;url&gt;</span>
        </div>
      </Panel>
    );
  }

  return (
    <Panel title="Source Distribution" icon="◉" accentColor="var(--cyan)">
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data.sources}
            dataKey="count"
            nameKey="source_type"
            cx="50%"
            cy="50%"
            outerRadius={80}
            innerRadius={40}
            paddingAngle={2}
            label={({ source_type, percent }) =>
              `${source_type} ${(percent * 100).toFixed(0)}%`
            }
            labelLine={false}
          >
            {data.sources.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#1a1a2e', border: '1px solid var(--border)', borderRadius: 6 }}
            itemStyle={{ color: 'var(--text-primary)' }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: 'var(--text-secondary)' }}
          />
        </PieChart>
      </ResponsiveContainer>
      <div style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>
        {data.total} total extractions
      </div>
    </Panel>
  );
}
