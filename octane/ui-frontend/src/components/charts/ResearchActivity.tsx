import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface ActivityData {
  activity: { date: string; count: number }[];
  total: number;
  days: number;
}

export function ResearchActivity() {
  const { data, loading } = useApi<ActivityData>('/api/charts/research-activity', 60000);

  if (loading) return <Panel title="Research Activity" icon="▥">Loading…</Panel>;
  if (!data?.activity?.length) {
    return (
      <Panel title="Research Activity" icon="▥">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No research activity yet.<br />
          <span style={{ fontSize: '11px' }}>Try: octane research extract &lt;url&gt;</span>
        </div>
      </Panel>
    );
  }

  const chartData = data.activity.map(d => ({
    ...d,
    label: d.date.slice(5), // MM-DD
  }));

  return (
    <Panel title="Research Activity" icon="▥" accentColor="var(--cyan)">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fill: 'var(--text-dim)' }}
            axisLine={{ stroke: 'var(--border)' }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'var(--text-dim)' }}
            axisLine={{ stroke: 'var(--border)' }}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={{ background: '#1a1a2e', border: '1px solid var(--border)', borderRadius: 6 }}
            itemStyle={{ color: 'var(--text-primary)' }}
            labelFormatter={l => `Date: ${l}`}
          />
          <Bar dataKey="count" fill="#00d4ff" radius={[3, 3, 0, 0]} name="Extractions" />
        </BarChart>
      </ResponsiveContainer>
      <div style={{ textAlign: 'center', fontSize: '11px', color: 'var(--text-dim)', marginTop: '4px' }}>
        {data.total} extractions over {data.days} days
      </div>
    </Panel>
  );
}
