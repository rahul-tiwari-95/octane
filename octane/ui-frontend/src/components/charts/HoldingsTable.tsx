import { useApi } from '../../hooks/useApi';
import { Panel } from '../Panel';

interface PositionsData {
  positions: {
    ticker: string;
    quantity: number;
    avg_cost: number;
    current_price?: number;
    market_value?: number;
    cost_basis?: number;
    pnl?: number;
    pnl_pct?: number;
    weight?: number;
    sector?: string;
  }[];
  total_market_value: number;
  total_cost_basis: number;
  total_pnl: number;
  count: number;
}

type SortKey = 'ticker' | 'weight' | 'pnl' | 'pnl_pct' | 'market_value';

import { useState } from 'react';

export function HoldingsTable() {
  const { data, loading } = useApi<PositionsData>('/api/portfolio/positions', 30000);
  const [sortBy, setSortBy] = useState<SortKey>('weight');
  const [sortAsc, setSortAsc] = useState(false);

  if (loading) return <Panel title="Holdings" icon="◫">Loading…</Panel>;
  if (!data?.positions?.length) {
    return (
      <Panel title="Holdings" icon="◫">
        <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '40px 0' }}>
          No holdings.<br />
          <span style={{ fontSize: '11px' }}>Import: octane portfolio import ~/file.csv</span>
        </div>
      </Panel>
    );
  }

  const sorted = [...data.positions].sort((a, b) => {
    const av = (a as any)[sortBy] ?? 0;
    const bv = (b as any)[sortBy] ?? 0;
    if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av - bv : bv - av;
  });

  const handleSort = (key: SortKey) => {
    if (sortBy === key) setSortAsc(!sortAsc);
    else { setSortBy(key); setSortAsc(false); }
  };

  const thStyle = (key: SortKey): React.CSSProperties => ({
    padding: '6px 8px',
    textAlign: key === 'ticker' ? 'left' : 'right',
    cursor: 'pointer',
    color: sortBy === key ? 'var(--green)' : 'var(--text-secondary)',
    fontSize: '10px',
    letterSpacing: '1px',
    textTransform: 'uppercase',
    borderBottom: '1px solid var(--border)',
    whiteSpace: 'nowrap',
    userSelect: 'none',
  });

  return (
    <Panel title="Holdings" icon="◫">
      <div style={{ overflowX: 'auto' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '12px',
          fontFamily: 'var(--font-mono)',
        }}>
          <thead>
            <tr>
              <th style={thStyle('ticker')} onClick={() => handleSort('ticker')}>
                Ticker {sortBy === 'ticker' ? (sortAsc ? '↑' : '↓') : ''}
              </th>
              <th style={{ ...thStyle('weight'), textAlign: 'right' }}>Shares</th>
              <th style={thStyle('market_value')} onClick={() => handleSort('market_value')}>
                Value {sortBy === 'market_value' ? (sortAsc ? '↑' : '↓') : ''}
              </th>
              <th style={thStyle('pnl')} onClick={() => handleSort('pnl')}>
                P&L {sortBy === 'pnl' ? (sortAsc ? '↑' : '↓') : ''}
              </th>
              <th style={thStyle('pnl_pct')} onClick={() => handleSort('pnl_pct')}>
                % {sortBy === 'pnl_pct' ? (sortAsc ? '↑' : '↓') : ''}
              </th>
              <th style={thStyle('weight')} onClick={() => handleSort('weight')}>
                Weight {sortBy === 'weight' ? (sortAsc ? '↑' : '↓') : ''}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(p => {
              const pnlColor = (p.pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)';
              return (
                <tr key={p.ticker} style={{ borderBottom: '1px solid var(--bg-hover)' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {p.ticker}
                    {p.sector && (
                      <span style={{ color: 'var(--text-dim)', fontSize: '9px', marginLeft: '6px' }}>
                        {p.sector}
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '5px 8px', textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {p.quantity.toLocaleString()}
                  </td>
                  <td style={{ padding: '5px 8px', textAlign: 'right', color: 'var(--text-primary)' }}>
                    {p.market_value != null ? `$${p.market_value.toLocaleString()}` : '—'}
                  </td>
                  <td style={{ padding: '5px 8px', textAlign: 'right', color: pnlColor }}>
                    {p.pnl != null ? `${p.pnl >= 0 ? '+' : ''}$${p.pnl.toLocaleString()}` : '—'}
                  </td>
                  <td style={{ padding: '5px 8px', textAlign: 'right', color: pnlColor }}>
                    {p.pnl_pct != null ? `${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct.toFixed(1)}%` : '—'}
                  </td>
                  <td style={{ padding: '5px 8px', textAlign: 'right', color: 'var(--text-secondary)' }}>
                    {p.weight != null ? `${p.weight.toFixed(1)}%` : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr style={{ borderTop: '1px solid var(--border)' }}>
              <td style={{ padding: '6px 8px', fontWeight: 700, color: 'var(--text-primary)' }}>
                TOTAL ({data.count})
              </td>
              <td />
              <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 700, color: 'var(--text-primary)' }}>
                ${data.total_market_value.toLocaleString()}
              </td>
              <td style={{
                padding: '6px 8px',
                textAlign: 'right',
                fontWeight: 700,
                color: data.total_pnl >= 0 ? 'var(--green)' : 'var(--red)',
              }}>
                {data.total_pnl >= 0 ? '+' : ''}${data.total_pnl.toLocaleString()}
              </td>
              <td />
              <td style={{ padding: '6px 8px', textAlign: 'right', color: 'var(--text-dim)' }}>100%</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </Panel>
  );
}
