import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface DividendData {
  dividends: { month: string; income: number }[];
  total_annual: number;
}

export function DividendBar() {
  const { data, loading } = useApi<DividendData>('/api/portfolio/dividends', 60000);

  if (loading) return <Panel title="Dividends" icon="💰" accentColor="var(--amber)">Loading…</Panel>;
  if (!data?.dividends?.length) {
    return (
      <Panel title="Dividends" icon="💰" accentColor="var(--amber)">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No dividend data yet.
        </div>
      </Panel>
    );
  }

  return (
    <Panel title="Dividends" icon="💰" accentColor="var(--amber)">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data.dividends}>
          <XAxis
            dataKey="month"
            tick={{ fill: 'var(--text-dim)', fontSize: 9 }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
            tickFormatter={(v: string) => v.slice(5)}  // MM only
          />
          <YAxis
            tick={{ fill: 'var(--text-dim)', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `$${v}`}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-panel)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-primary)',
              fontSize: '12px',
            }}
            formatter={(value: number) => [`$${value.toFixed(2)}`, 'Income']}
          />
          <Bar dataKey="income" radius={[3, 3, 0, 0]}>
            {data.dividends.map((_, i) => (
              <Cell key={i} fill="#ffaa00" fillOpacity={0.7 + (i % 3) * 0.1} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{
        textAlign: 'center',
        color: 'var(--amber)',
        fontSize: '11px',
        marginTop: '4px',
      }}>
        Annual: ${data.total_annual.toLocaleString()}
      </div>
    </Panel>
  );
}
