import { SystemVitals } from '../components/SystemVitals';
import { ModelPanel } from '../components/ModelPanel';
import { EventStream } from '../components/EventStream';
import { QueryTimeline } from '../components/QueryTimeline';
import { Globe } from '../components/Globe';

export function Dashboard() {
  return (
    <div className="dashboard-grid">
      {/* Hero row: vitals — globe — models */}
      <div className="hero-row">
        <div className="cell cell-vitals">
          <SystemVitals />
        </div>
        <div className="cell cell-globe">
          <Globe />
        </div>
        <div className="cell cell-models">
          <ModelPanel />
        </div>
      </div>
      {/* Bottom row: events + timeline */}
      <div className="bottom-row">
        <div className="cell cell-events">
          <EventStream />
        </div>
        <div className="cell cell-timeline">
          <QueryTimeline />
        </div>
      </div>
    </div>
  );
}
