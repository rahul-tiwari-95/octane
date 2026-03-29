import { ReactNode } from 'react';

interface PanelProps {
  title: string;
  icon?: string;
  children: ReactNode;
  className?: string;
  accentColor?: string;
}

export function Panel({ title, icon, children, className = '', accentColor = 'var(--green)' }: PanelProps) {
  return (
    <div className={`panel ${className}`} style={{
      background: 'var(--bg-panel)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius)',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        padding: '8px 12px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '11px',
        textTransform: 'uppercase',
        letterSpacing: '1.5px',
        color: 'var(--text-secondary)',
      }}>
        {icon && <span style={{ color: accentColor, fontSize: '13px' }}>{icon}</span>}
        <span>{title}</span>
        <span style={{
          marginLeft: 'auto',
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: accentColor,
          animation: 'pulse-glow 2s ease-in-out infinite',
        }} />
      </div>
      <div style={{ padding: '12px', flex: 1, overflow: 'auto' }}>
        {children}
      </div>
    </div>
  );
}
