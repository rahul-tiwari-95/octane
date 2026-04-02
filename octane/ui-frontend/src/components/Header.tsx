import { NavLink } from 'react-router-dom';
import { useEvents } from '../context/EventsContext';

const navStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
  color: isActive ? 'var(--green)' : 'var(--text-dim)',
  textDecoration: 'none',
  fontSize: '11px',
  letterSpacing: '1.5px',
  textTransform: 'uppercase',
  padding: '4px 12px',
  borderBottom: isActive ? '1px solid var(--green)' : '1px solid transparent',
  transition: 'all 0.2s ease',
});

export function Header() {
  const { connected } = useEvents();
  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      height: '42px',
      background: 'var(--bg-panel)',
      borderBottom: '1px solid var(--border)',
      flexShrink: 0,
    }}>
      {/* Left — branding */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span style={{
          color: 'var(--green)',
          fontWeight: 700,
          fontSize: '14px',
          letterSpacing: '3px',
          fontFamily: 'var(--font-mono)',
        }}>
          OCTANE
        </span>
        <span className="header-subtitle" style={{
          color: 'var(--text-dim)',
          fontSize: '10px',
          letterSpacing: '2px',
        }}>
          MISSION CONTROL
        </span>
      </div>

      {/* Center — nav */}
      <nav style={{ display: 'flex', gap: '4px' }}>
        <NavLink to="/" style={navStyle} end>
          Dashboard
        </NavLink>
        <NavLink to="/portfolio" style={navStyle}>
          Portfolio
        </NavLink>
        <NavLink to="/terminal" style={navStyle}>
          Terminal
        </NavLink>
      </nav>

      {/* Right — status */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        fontSize: '10px',
        color: 'var(--text-dim)',
      }}>
        <span style={{
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
        }}>
          <span style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: connected ? 'var(--green)' : 'var(--red)',
            boxShadow: connected
              ? '0 0 6px var(--green)'
              : '0 0 6px var(--red)',
            animation: connected ? 'pulse-glow 2s ease-in-out infinite' : 'none',
          }} />
          {connected ? 'LIVE' : 'OFFLINE'}
        </span>
        <span className="header-host" style={{
          color: 'var(--green-dim)',
          letterSpacing: '1px',
        }}>
          octane.local
        </span>
      </div>
    </header>
  );
}
