/**
 * Single shared WebSocket connection for Synapse events.
 *
 * Wraps the entire app so that every component that needs live events
 * simply calls `useEvents()` — no duplicate connections.
 */
import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import type { SynapseEvent } from '../hooks/useWebSocket';

interface EventsContextValue {
  events: SynapseEvent[];
  connected: boolean;
}

const EventsCtx = createContext<EventsContextValue>({
  events: [],
  connected: false,
});

export function useEvents() {
  return useContext(EventsCtx);
}

export function EventsProvider({ children }: { children: ReactNode }) {
  const [events, setEvents] = useState<SynapseEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    // Guard against duplicate connections (React StrictMode double-mount)
    const cur = wsRef.current;
    if (cur && (cur.readyState === WebSocket.OPEN || cur.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/events`);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      reconnectRef.current = setTimeout(connect, 3000);
    };
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as SynapseEvent;
        setEvents((prev) => [...prev.slice(-200), event]);
      } catch {
        // non-JSON payload, ignore
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  return (
    <EventsCtx.Provider value={{ events, connected }}>
      {children}
    </EventsCtx.Provider>
  );
}
