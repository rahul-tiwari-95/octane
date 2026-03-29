import { useRef, useMemo, useCallback, useState, useEffect } from 'react';
import GlobeGL from 'react-globe.gl';
import { MeshPhongMaterial } from 'three';
import { useEvents } from '../context/EventsContext';
import type { SynapseEvent } from '../hooks/useWebSocket';

// ── Coordinate mappings ─────────────────────────────────────────────────────
// Spread domains across real-world cities to avoid single-cluster look

const DOMAIN_COORDS: Record<string, [number, number]> = {
  // North America
  'arxiv.org': [42.36, -71.09],
  'github.com': [37.78, -122.39],
  'wikipedia.org': [38.90, -77.04],
  'youtube.com': [37.42, -122.08],
  'google.com': [37.42, -122.08],
  'reddit.com': [37.39, -122.08],
  'twitter.com': [37.78, -122.39],
  'x.com': [37.78, -122.39],
  'nytimes.com': [40.76, -73.99],
  'wsj.com': [40.71, -74.01],
  'bloomberg.com': [40.76, -73.97],
  'stackoverflow.com': [40.69, -74.04],
  'hn': [37.76, -122.42],
  'openai.com': [37.78, -122.41],
  'anthropic.com': [37.76, -122.39],
  'aws.amazon.com': [47.62, -122.35],
  'microsoft.com': [47.64, -122.13],
  'apple.com': [37.33, -122.03],
  'meta.com': [37.48, -122.15],
  // Europe
  'bbc.com': [51.51, -0.13],
  'bbc.co.uk': [51.52, -0.12],
  'reuters.com': [51.51, -0.10],
  'ft.com': [51.51, -0.08],
  'nature.com': [51.53, -0.13],
  'sciencedirect.com': [52.34, 4.89],
  'springer.com': [52.52, 13.41],
  'huggingface.co': [48.86, 2.35],
  // Asia
  'arxiv.org': [42.36, -71.09],
  'ieee.org': [40.05, -75.15],
  'baidu.com': [39.91, 116.39],
  'tencent.com': [22.54, 114.06],
  'nikkei.com': [35.68, 139.69],
};

// Spread TLDs more geographically
const TLD_COORDS: Record<string, [number, number]> = {
  '.com': [37.0, -95.0],
  '.org': [39.0, -77.0],
  '.edu': [42.36, -71.09],
  '.gov': [38.89, -77.03],
  '.co.uk': [51.51, -0.13],
  '.de': [52.52, 13.41],
  '.fr': [48.86, 2.35],
  '.jp': [35.68, 139.69],
  '.cn': [39.91, 116.39],
  '.in': [28.61, 77.21],
  '.au': [-33.87, 151.21],
  '.ca': [45.42, -75.69],
  '.io': [51.49, -0.01],    // London (many UK companies)
  '.ai': [18.22, -63.05],   // Anguilla
  '.kr': [37.57, 126.98],
  '.br': [-23.55, -46.63],
  '.mx': [19.43, -99.13],
  '.ru': [55.76, 37.62],
  '.se': [59.33, 18.07],
  '.nl': [52.37, 4.90],
  '.it': [41.90, 12.50],
};

// Category → color mapping for varied point colors
const EVENT_COLORS: Record<string, string> = {
  web: '#00ff88',
  research: '#00ccff',
  extract: '#ff9f43',
  inference: '#a55eea',
  agent: '#fd79a8',
  memory: '#ffeaa7',
  finance: '#55efc4',
  system: '#636e72',
};

// ── Node locations for ambient grid points (world tech/research hubs) ───────
const AMBIENT_NODES: Array<{ lat: number; lng: number; label: string }> = [
  { lat: 37.78, lng: -122.39, label: 'San Francisco' },
  { lat: 40.71, lng: -74.01, label: 'New York' },
  { lat: 51.51, lng: -0.13, label: 'London' },
  { lat: 48.86, lng: 2.35, label: 'Paris' },
  { lat: 52.52, lng: 13.41, label: 'Berlin' },
  { lat: 35.68, lng: 139.69, label: 'Tokyo' },
  { lat: 22.54, lng: 114.06, label: 'Shenzhen' },
  { lat: 39.91, lng: 116.39, label: 'Beijing' },
  { lat: 12.97, lng: 77.59, label: 'Bengaluru' },
  { lat: 1.35, lng: 103.82, label: 'Singapore' },
  { lat: -33.87, lng: 151.21, label: 'Sydney' },
  { lat: 37.57, lng: 126.98, label: 'Seoul' },
  { lat: 47.62, lng: -122.35, label: 'Seattle' },
  { lat: 30.27, lng: -97.74, label: 'Austin' },
  { lat: 43.65, lng: -79.38, label: 'Toronto' },
  { lat: 55.76, lng: 37.62, label: 'Moscow' },
  { lat: -23.55, lng: -46.63, label: 'São Paulo' },
  { lat: 25.20, lng: 55.27, label: 'Dubai' },
  { lat: 59.33, lng: 18.07, label: 'Stockholm' },
  { lat: 52.37, lng: 4.90, label: 'Amsterdam' },
];

interface DataPoint {
  lat: number;
  lng: number;
  label: string;
  size: number;
  color: string;
  altitude: number;
  ts: number;
}

interface RingDatum {
  lat: number;
  lng: number;
  maxR: number;
  propagationSpeed: number;
  repeatPeriod: number;
  color: string;
}

function extractDomain(url: string): string {
  try {
    const u = new URL(url.startsWith('http') ? url : `https://${url}`);
    return u.hostname.replace('www.', '');
  } catch {
    return url;
  }
}

function domainToCoords(domain: string): [number, number] {
  if (DOMAIN_COORDS[domain]) return DOMAIN_COORDS[domain];
  for (const [tld, coords] of Object.entries(TLD_COORDS)) {
    if (domain.endsWith(tld)) {
      return [
        coords[0] + (Math.random() - 0.5) * 8,
        coords[1] + (Math.random() - 0.5) * 8,
      ];
    }
  }
  // Place unknown domains at a random world city for visual spread
  const node = AMBIENT_NODES[Math.floor(Math.random() * AMBIENT_NODES.length)];
  return [
    node.lat + (Math.random() - 0.5) * 10,
    node.lng + (Math.random() - 0.5) * 10,
  ];
}

function categoriseEvent(event: SynapseEvent): string {
  const t = event.event_type.toLowerCase();
  if (t.includes('web') || t.includes('scrape') || t.includes('fetch')) return 'web';
  if (t.includes('research') || t.includes('finding')) return 'research';
  if (t.includes('extract') || t.includes('pdf') || t.includes('youtube')) return 'extract';
  if (t.includes('infer') || t.includes('llm') || t.includes('model') || t.includes('bodega')) return 'inference';
  if (t.includes('agent') || t.includes('task') || t.includes('dag')) return 'agent';
  if (t.includes('memory') || t.includes('recall') || t.includes('embed')) return 'memory';
  if (t.includes('portfolio') || t.includes('finance') || t.includes('price')) return 'finance';
  return 'system';
}

function eventToPoints(event: SynapseEvent): DataPoint[] {
  const points: DataPoint[] = [];
  const payload = event.payload || {};
  const category = categoriseEvent(event);
  const color = EVENT_COLORS[category] || '#636e72';

  // Extract URLs from web_search events
  const urls: string[] = payload.urls || payload.sources || [];
  if (payload.url) urls.push(payload.url);
  if (payload.source_url) urls.push(payload.source_url);

  for (const url of urls) {
    const domain = extractDomain(url);
    const [lat, lng] = domainToCoords(domain);
    points.push({
      lat, lng,
      label: `${domain} (${category})`,
      size: 0.35 + Math.random() * 0.2,
      color,
      altitude: 0.005 + Math.random() * 0.015,
      ts: Date.now(),
    });
  }

  // Generate a point for ANY event, not just web/research
  if (points.length === 0) {
    const node = AMBIENT_NODES[Math.floor(Math.random() * AMBIENT_NODES.length)];
    points.push({
      lat: node.lat + (Math.random() - 0.5) * 6,
      lng: node.lng + (Math.random() - 0.5) * 6,
      label: `${event.event_type} (${category})`,
      size: 0.25 + Math.random() * 0.15,
      color,
      altitude: 0.003 + Math.random() * 0.01,
      ts: Date.now(),
    });
  }

  return points;
}

export function Globe() {
  const globeRef = useRef<any>(null);
  const { events } = useEvents();
  const [globeMaterial] = useState(
    () => new MeshPhongMaterial({ color: '#080818', transparent: true, opacity: 0.9 }),
  );

  // Auto-rotate
  useEffect(() => {
    if (globeRef.current) {
      const controls = globeRef.current.controls();
      if (controls) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.3;
      }
    }
  }, []);

  // Always show ambient baseline nodes so globe never looks empty
  const ambientPoints = useMemo<DataPoint[]>(() => {
    return AMBIENT_NODES.map((n) => ({
      lat: n.lat,
      lng: n.lng,
      label: n.label,
      size: 0.15,
      color: '#1e3a2e',
      altitude: 0.001,
      ts: 0,
    }));
  }, []);

  // Convert recent events to data points
  const eventPoints = useMemo(() => {
    const points: DataPoint[] = [];
    for (const event of events.slice(-80)) {
      points.push(...eventToPoints(event));
    }
    return points.slice(-150);
  }, [events]);

  const dataPoints = useMemo(
    () => [...ambientPoints, ...eventPoints],
    [ambientPoints, eventPoints],
  );

  // Rings — pulsing circles at the 5 most recent event locations
  const ringsData = useMemo<RingDatum[]>(() => {
    return eventPoints.slice(-5).map((p) => ({
      lat: p.lat,
      lng: p.lng,
      maxR: 3,
      propagationSpeed: 2,
      repeatPeriod: 1200,
      color: p.color,
    }));
  }, [eventPoints]);

  // Arcs — connect recent event points to each other (mesh network look)
  const arcsData = useMemo(() => {
    const recent = eventPoints.slice(-12);
    const arcs: Array<{
      startLat: number; startLng: number;
      endLat: number; endLng: number;
      color: [string, string];
    }> = [];
    for (let i = 1; i < recent.length; i++) {
      const prev = recent[i - 1];
      const cur = recent[i];
      arcs.push({
        startLat: prev.lat,
        startLng: prev.lng,
        endLat: cur.lat,
        endLng: cur.lng,
        color: [
          `${cur.color}99`,
          `${prev.color}33`,
        ],
      });
    }
    return arcs;
  }, [eventPoints]);

  const pointLabel = useCallback((d: object) => {
    const point = d as DataPoint;
    return `<span style="color: ${point.color}; font-family: monospace; font-size: 11px">${point.label}</span>`;
  }, []);

  // Category summary for HUD
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const event of events.slice(-80)) {
      const cat = categoriseEvent(event);
      counts[cat] = (counts[cat] || 0) + 1;
    }
    return counts;
  }, [events]);

  return (
    <div style={{
      width: '100%',
      height: '100%',
      position: 'relative',
      overflow: 'hidden',
      borderRadius: '6px',
    }}>
      <GlobeGL
        ref={globeRef}
        width={480}
        height={440}
        backgroundColor="rgba(0,0,0,0)"
        globeImageUrl=""
        showGlobe={true}
        showAtmosphere={true}
        atmosphereColor="#00ff8844"
        atmosphereAltitude={0.18}
        globeMaterial={globeMaterial}
        hexPolygonsData={[]}
        pointsData={dataPoints}
        pointLat="lat"
        pointLng="lng"
        pointAltitude="altitude"
        pointRadius="size"
        pointColor="color"
        pointLabel={pointLabel}
        arcsData={arcsData}
        arcStartLat="startLat"
        arcStartLng="startLng"
        arcEndLat="endLat"
        arcEndLng="endLng"
        arcColor="color"
        arcDashLength={0.5}
        arcDashGap={0.3}
        arcDashAnimateTime={2000}
        arcStroke={0.4}
        ringsData={ringsData}
        ringLat="lat"
        ringLng="lng"
        ringMaxRadius="maxR"
        ringPropagationSpeed="propagationSpeed"
        ringRepeatPeriod="repeatPeriod"
        ringColor={() => (t: number) => `rgba(0, 255, 136, ${1 - t})`}
        animateIn={true}
        enablePointerInteraction={true}
      />
      {/* HUD overlay — top-left mini legend */}
      <div style={{
        position: 'absolute',
        top: '6px',
        left: '8px',
        fontSize: '8px',
        lineHeight: '13px',
        color: 'var(--text-dim)',
        fontFamily: 'var(--font-mono)',
        pointerEvents: 'none',
        opacity: 0.7,
      }}>
        {Object.entries(categoryCounts).slice(0, 5).map(([cat, count]) => (
          <div key={cat}>
            <span style={{ color: EVENT_COLORS[cat] || '#636e72' }}>●</span>{' '}
            {cat} {count}
          </div>
        ))}
      </div>
      {/* HUD overlay — bottom status */}
      <div style={{
        position: 'absolute',
        bottom: '4px',
        left: '8px',
        fontSize: '9px',
        color: 'var(--text-dim)',
        fontFamily: 'var(--font-mono)',
        opacity: 0.6,
      }}>
        {eventPoints.length} events · {AMBIENT_NODES.length} nodes
      </div>
    </div>
  );
}
