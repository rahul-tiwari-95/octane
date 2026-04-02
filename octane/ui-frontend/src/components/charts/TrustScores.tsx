import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface TrustData {
  distribution: { tier: string; count: number }[];
  total: number;
}

const TIER_COLORS: Record<string, string> = {
  high: '#00ff88',
  medium: '#ffaa00',
  low: '#ff4444',
  unknown: '#555577',
};

export function TrustScores() {
  const { data, loading } = useApi<TrustData>('/api/charts/trust-scores', 60000);

  if (loading) return <Panel title="Trust Scores" icon="⛊">Loading…</Panel>;
  if (!data?.distribution?.length) {
    return (
      <Panel title="Trust Scores" icon="⛊">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No scored documents yet.
        </div>
      </Panel>
    );
  }

  return (
    <Panel title="Trust Scores" icon="⛊" accentColor="var(--green)">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data.distribution} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <XAxis
            dataKey="tier"
            tick={{ fontSize: 11, fill: 'var(--text-secondary)', textTransform: 'uppercase' }}
            axisLine={{ stroke: 'var(--border)' }}
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'var(--text-dim)' }}
            axisLine={{ stroke: 'var(--border)' }}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={{ background: '#1a1a2e', border: '1px solid var(--border)', borderRadius: 6 }}
            itemStyle={{ color: 'var(--text-primary)' }}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Documents">
            {data.distribution.map((d, i) => (
              <Cell key={i} fill={TIER_COLORS[d.tier] ?? '#555577'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>
        {data.total} scored documents
      </div>
    </Panel>
  );
}
