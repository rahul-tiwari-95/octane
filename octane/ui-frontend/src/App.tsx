import { BrowserRouter, useLocation } from 'react-router-dom';
import { Header } from './components/Header';
import { Dashboard } from './pages/Dashboard';
import { TerminalPage } from './pages/TerminalPage';
import { EventsProvider } from './context/EventsContext';
import './App.css';

function AppContent() {
  const location = useLocation();
  const onTerminal = location.pathname.startsWith('/terminal');

  return (
    <div className="app-shell">
      <Header />
      <main className="app-main">
        {/* Dashboard: mount/unmount with route */}
        {!onTerminal && <Dashboard />}

        {/* Terminal: always mounted, CSS show/hide for session persistence */}
        <div style={{
          display: onTerminal ? 'flex' : 'none',
          flexDirection: 'column',
          height: '100%',
          flex: 1,
        }}>
          <TerminalPage visible={onTerminal} />
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <EventsProvider>
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </EventsProvider>
  );
}

export default App;
