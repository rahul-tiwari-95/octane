import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

const COLORS = [
  '#00d4ff', '#00ff88', '#ff9900', '#ff4488',
  '#cc88ff', '#ffdd00', '#44aaff', '#ff6644',
  '#88ffcc', '#dd44ff',
];

interface Allocation {
  ticker: string;
  value: number;
  weight: number;
  sector: string;
}

interface AllocData {
  allocations: Allocation[];
  total: number;
}

export function AllocationDonut() {
  const { data, loading } = useApi<AllocData>('/api/portfolio/allocation', 30000);

  if (loading) return <Panel title="Allocation" icon="◉">Loading…</Panel>;
  if (!data?.allocations?.length) {
    return (
      <Panel title="Allocation" icon="◉">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No portfolio data.<br />
          <span style={{ fontSize: '11px' }}>Import with: octane portfolio import</span>
        </div>
      </Panel>
    );
  }

  const chartData = data.allocations.map(a => ({
    name: a.ticker,
    value: a.value,
    weight: a.weight,
  }));

  return (
    <Panel title="Allocation" icon="◉">
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius="55%"
            outerRadius="85%"
            paddingAngle={2}
            stroke="var(--bg-primary)"
            strokeWidth={2}
          >
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: 'var(--bg-panel)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-primary)',
              fontSize: '12px',
            }}
            formatter={(value: number, name: string) => [
              `$${value.toLocaleString()}`,
              name,
            ]}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            formatter={(value: string) => (
              <span style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
      <div style={{
        textAlign: 'center',
        color: 'var(--text-dim)',
        fontSize: '11px',
        marginTop: '-8px',
      }}>
        Total: ${data.total.toLocaleString()}
      </div>
    </Panel>
  );
}
