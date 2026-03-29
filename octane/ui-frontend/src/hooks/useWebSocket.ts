import { useEffect, useRef, useState, useCallback } from 'react';

export interface SynapseEvent {
  event_id: string;
  correlation_id: string;
  event_type: string;
  source: string;
  target: string;
  payload: Record<string, unknown>;
  error: string;
  duration_ms: number;
  timestamp: string;
}

export function useWebSocket(path: string) {
  const [events, setEvents] = useState<SynapseEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const ws = new WebSocket(`${protocol}//${host}${path}`);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      reconnectRef.current = setTimeout(connect, 3000);
    };
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as SynapseEvent;
        setEvents((prev) => [...prev.slice(-200), event]);
      } catch {
        // binary data from terminal, ignore
      }
    };

    wsRef.current = ws;
  }, [path]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { events, connected };
}

export type { SynapseEvent };
