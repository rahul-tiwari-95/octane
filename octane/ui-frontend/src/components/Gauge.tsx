interface GaugeProps {
  label: string;
  value: number;     // 0-100
  color?: string;
  suffix?: string;
}

export function Gauge({ label, value, color = 'var(--green)', suffix = '%' }: GaugeProps) {
  const barWidth = Math.min(Math.max(value, 0), 100);
  const barColor = value > 90 ? 'var(--red)' : value > 70 ? 'var(--amber)' : color;

  return (
    <div style={{ marginBottom: '8px' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '11px',
        marginBottom: '3px',
      }}>
        <span style={{ color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px' }}>
          {label}
        </span>
        <span style={{ color: barColor, fontWeight: 600 }}>
          {Math.round(value)}{suffix}
        </span>
      </div>
      <div style={{
        height: '4px',
        background: 'var(--bg-secondary)',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${barWidth}%`,
          background: barColor,
          borderRadius: '2px',
          transition: 'width 0.5s ease, background 0.3s ease',
          boxShadow: `0 0 8px ${barColor}40`,
        }} />
      </div>
    </div>
  );
}
