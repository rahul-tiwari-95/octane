import { AllocationDonut } from '../components/charts/AllocationDonut';
import { SectorRadar } from '../components/charts/SectorRadar';
import { CorrelationHeatmap } from '../components/charts/CorrelationHeatmap';
import { NetWorthTimeline } from '../components/charts/NetWorthTimeline';
import { DividendBar } from '../components/charts/DividendBar';
import { HoldingsTable } from '../components/charts/HoldingsTable';
import { SourceDistribution } from '../components/charts/SourceDistribution';
import { ResearchActivity } from '../components/charts/ResearchActivity';
import { TrustScores } from '../components/charts/TrustScores';

export function PortfolioPage() {
  return (
    <div className="portfolio-grid">
      {/* Row 1: Allocation + Sector + Correlation */}
      <div className="pf-row pf-row-top">
        <div className="pf-cell pf-alloc">
          <AllocationDonut />
        </div>
        <div className="pf-cell pf-sector">
          <SectorRadar />
        </div>
        <div className="pf-cell pf-corr">
          <CorrelationHeatmap />
        </div>
      </div>

      {/* Row 2: Net Worth + Dividends */}
      <div className="pf-row pf-row-mid">
        <div className="pf-cell pf-nw">
          <NetWorthTimeline />
        </div>
        <div className="pf-cell pf-div">
          <DividendBar />
        </div>
      </div>

      {/* Row 3: Holdings */}
      <div className="pf-row pf-row-holdings">
        <div className="pf-cell pf-holdings">
          <HoldingsTable />
        </div>
      </div>

      {/* Row 4: Research analytics */}
      <div className="pf-row pf-row-research">
        <div className="pf-cell pf-src">
          <SourceDistribution />
        </div>
        <div className="pf-cell pf-activity">
          <ResearchActivity />
        </div>
        <div className="pf-cell pf-trust">
          <TrustScores />
        </div>
      </div>
    </div>
  );
}
