import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface Snapshot {
  date: string;
  total: number;
  invested: number | null;
}

interface NetWorthData {
  snapshots: Snapshot[];
  count: number;
}

export function NetWorthTimeline() {
  const { data, loading } = useApi<NetWorthData>('/api/portfolio/net-worth', 60000);

  if (loading) return <Panel title="Net Worth" icon="📈" accentColor="var(--green)">Loading…</Panel>;
  if (!data?.snapshots?.length) {
    return (
      <Panel title="Net Worth" icon="📈" accentColor="var(--green)">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No snapshots yet.<br />
          <span style={{ fontSize: '11px' }}>Run: octane portfolio snapshot</span>
        </div>
      </Panel>
    );
  }

  const chartData = data.snapshots.map(s => ({
    date: s.date.slice(5),  // MM-DD
    total: s.total,
    invested: s.invested,
  }));

  return (
    <Panel title="Net Worth" icon="📈" accentColor="var(--green)">
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="nwGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00ff88" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#00ff88" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tick={{ fill: 'var(--text-dim)', fontSize: 9 }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: 'var(--text-dim)', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-panel)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-primary)',
              fontSize: '12px',
            }}
            formatter={(value: number) => [`$${value.toLocaleString()}`, 'Total']}
          />
          <Area
            type="monotone"
            dataKey="total"
            stroke="#00ff88"
            fill="url(#nwGrad)"
            strokeWidth={2}
          />
          {chartData[0]?.invested != null && (
            <Line
              type="monotone"
              dataKey="invested"
              stroke="var(--text-dim)"
              strokeDasharray="4 4"
              strokeWidth={1}
              dot={false}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </Panel>
  );
}
