import { useEffect, useState } from 'react';
import { Documents, DocumentConsistency } from '../components/Documents';
import { CaseOverview } from '../components/CaseOverview';
import { DemoFlow } from './DemoFlow';
import {
  AgentRecommendation,
  Audit,
  EvidenceDrawer,
  KnowledgeEvidence,
  SimilarClaims,
} from '../components/Evidence';
import { Anomaly, ClaimHistory, Investigation, Triage } from '../components/Intelligence';
import { VisualConsistency, VisualEvidence } from '../components/Vision';
import { Badge, FactList, Metric, Section, anchor, date, humanize, money } from '../presentation';
import type { ReviewView } from '../types';

type ViewTab = 'overview' | 'evidence' | 'technical';
type GroupId = 'documents' | 'photos' | 'policy' | 'history' | 'recommendation';
const tabs: { id: ViewTab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'technical', label: 'Technical details' },
];
const groups: { id: GroupId; label: string; description: string }[] = [
  {
    id: 'documents',
    label: 'Documents',
    description: 'Files, extracted fields and document comparisons',
  },
  {
    id: 'photos',
    label: 'Photos',
    description: 'Original images, recorded observations and visual consistency',
  },
  {
    id: 'policy',
    label: 'Policy & procedure',
    description: 'The source excerpts behind the recorded guidance',
  },
  {
    id: 'history',
    label: 'History & similar claims',
    description: 'Earlier incidents and comparative context',
  },
  {
    id: 'recommendation',
    label: 'Recommendation & signals',
    description: 'The full recommendation, uncertainties and review signals',
  },
];

function destination(section: string): { tab: ViewTab; group?: GroupId } {
  if (['documents', 'consistency'].includes(section))
    return { tab: 'evidence', group: 'documents' };
  if (['vision', 'visual_consistency'].includes(section))
    return { tab: 'evidence', group: 'photos' };
  if (section === 'knowledge') return { tab: 'evidence', group: 'policy' };
  if (['history', 'similar_claims'].includes(section)) return { tab: 'evidence', group: 'history' };
  if (['agent', 'investigation'].includes(section))
    return { tab: 'evidence', group: 'recommendation' };
  if (['triage', 'anomaly', 'audit', 'priority-details'].includes(section))
    return { tab: 'technical' };
  return { tab: 'overview' };
}

export function ReviewConsole({
  view,
  scenarioTitle,
  onBack,
}: {
  view: ReviewView;
  scenarioTitle?: string;
  onBack?: () => void;
}) {
  const [activeTab, setActiveTab] = useState<ViewTab>('overview');
  const [activeGroup, setActiveGroup] = useState<GroupId | null>(null);
  const [evidenceId, setEvidenceId] = useState<string | null>(null);
  const [focusRequest, setFocusRequest] = useState<{ section: string; evidenceId?: string } | null>(
    null,
  );
  const claim = view.claim;
  const investigation = view.investigation.status === 'AVAILABLE' ? view.investigation.data : null;

  function reveal(section: string, id?: string) {
    const target = destination(section);
    setActiveTab(target.tab);
    if (target.group) setActiveGroup(target.group);
    setFocusRequest({ section, evidenceId: id });
  }

  function openEvidence(id: string) {
    const source = view.evidence.find((item) => item.id === id);
    reveal(source?.section || 'overview', id);
  }

  useEffect(() => {
    if (!focusRequest) return;
    const target =
      (focusRequest.evidenceId && document.getElementById(anchor(focusRequest.evidenceId))) ||
      document.getElementById(focusRequest.section) ||
      document.getElementById(`review-panel-${activeTab}`);
    if (target) {
      target.scrollIntoView?.({ behavior: 'instant', block: 'center' });
      target.focus({ preventScroll: true });
      if (focusRequest.evidenceId) {
        target.classList.add('evidence-highlight');
        window.setTimeout(() => target.classList.remove('evidence-highlight'), 4000);
      }
    }
    // Reveal and focus the source before opening the modal, so its focus stays inside.
    if (focusRequest.evidenceId) setEvidenceId(focusRequest.evidenceId);
    setFocusRequest(null);
  }, [focusRequest, activeTab, activeGroup]);

  function groupContents(group: GroupId) {
    switch (group) {
      case 'documents':
        return (
          <>
            <Documents view={view} />
            <DocumentConsistency view={view} />
          </>
        );
      case 'photos':
        return (
          <>
            <VisualEvidence view={view} />
            <VisualConsistency view={view} />
          </>
        );
      case 'policy':
        return <KnowledgeEvidence view={view} />;
      case 'history':
        return (
          <>
            <ClaimHistory view={view} />
            <SimilarClaims view={view} />
          </>
        );
      case 'recommendation':
        return (
          <>
            <AgentRecommendation view={view} onEvidence={openEvidence} />
            <Investigation view={view} onEvidence={openEvidence} />
          </>
        );
    }
  }

  return (
    <>
      <header id="overview" className="review-case-heading" tabIndex={-1}>
        {onBack && (
          <button
            type="button"
            className="case-breadcrumb"
            aria-label="Explore other cases"
            onClick={onBack}
          >
            Explore other cases
          </button>
        )}
        <div className="case-title-row">
          <h1 id="review-case-title" tabIndex={-1}>
            {scenarioTitle || humanize(claim.claim_type)}
          </h1>
          {view.mode === 'demo' && <Badge tone="demo">Demo case</Badge>}
        </div>
        <p className="case-description">
          {claim.claim_id} · {humanize(claim.status)}
        </p>
        <dl className="case-summary-meta">
          <Metric
            label="Vehicle"
            value={`${claim.vehicle.make} ${claim.vehicle.model} · ${claim.vehicle.year}`}
          />
          <Metric
            label="Incident"
            value={`${date(claim.incident.date)} · ${claim.incident.location}`}
          />
          <Metric
            label="Repair estimate"
            value={money(claim.repair_estimate.amount, claim.repair_estimate.currency)}
          />
        </dl>
        <div className="case-priority-inline">
          <span>Review priority</span>
          <Badge tone={investigation?.review_priority || 'unavailable'}>
            {investigation?.review_priority || 'Not available'}
          </Badge>
          {investigation?.partial && <Badge tone="human">Partial assessment</Badge>}
          <span className="priority-explainer">
            Review prioritization — not a fraud probability.
          </span>
        </div>
      </header>
      <div className="human-banner">
        <span aria-hidden="true">◇</span>
        <strong>HUMAN REVIEW REQUIRED</strong>
        <span>AI-assisted guidance. Final decisions remain human.</span>
      </div>
      <div className="case-navigation" role="tablist" aria-label="Claim review views">
        {tabs.map((tab, index) => (
          <button
            type="button"
            role="tab"
            className="case-tab"
            id={`review-tab-${tab.id}`}
            key={tab.id}
            aria-selected={activeTab === tab.id}
            aria-controls={`review-panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => setActiveTab(tab.id)}
            onKeyDown={(event) => {
              let next: number;
              if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
              else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
              else if (event.key === 'Home') next = 0;
              else if (event.key === 'End') next = tabs.length - 1;
              else return;
              event.preventDefault();
              setActiveTab(tabs[next].id);
              document.getElementById(`review-tab-${tabs[next].id}`)?.focus();
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <section
        role="tabpanel"
        id={`review-panel-${activeTab}`}
        aria-labelledby={`review-tab-${activeTab}`}
        tabIndex={0}
      >
        {activeTab === 'overview' && (
          <>
            <CaseOverview view={view} onOpenDetails={reveal} />
            <DemoFlow />
          </>
        )}
        {activeTab === 'evidence' && (
          <div className="details-browser">
            <p className="section-help">
              Open one area to inspect its evidence. Full results remain available here.
            </p>
            {groups.map((group) => (
              <div className="evidence-group" key={group.id}>
                <button
                  type="button"
                  className="evidence-group-toggle"
                  aria-label={group.label}
                  aria-expanded={activeGroup === group.id}
                  aria-controls={`evidence-group-${group.id}`}
                  onClick={() => setActiveGroup(activeGroup === group.id ? null : group.id)}
                >
                  <span>
                    <strong className="evidence-group-label">{group.label}</strong>
                    <span className="evidence-group-description">{group.description}</span>
                  </span>
                  <span aria-hidden="true">{activeGroup === group.id ? '−' : '+'}</span>
                </button>
                {activeGroup === group.id && (
                  <div className="evidence-group-body" id={`evidence-group-${group.id}`}>
                    {groupContents(group.id)}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {activeTab === 'technical' && (
          <div className="technical-review">
            <p className="section-help">Optional: model outputs, versions and the audit trail.</p>
            <section
              className="panel"
              id="case-details"
              aria-labelledby="case-details-title"
              tabIndex={-1}
            >
              <h2 id="case-details-title">Case details</h2>
              <FactList
                facts={{
                  claim_id: claim.claim_id,
                  claim_type: claim.claim_type,
                  status: claim.status,
                  incident: claim.incident,
                  vehicle: claim.vehicle,
                  customer: claim.customer,
                  policy: claim.policy,
                  declared_damage: claim.declared_damage,
                  repair_estimate: claim.repair_estimate,
                }}
              />
            </section>
            <Section
              id="priority-details"
              title="Review prioritization"
              section={view.investigation}
            >
              {(assessment) => (
                <FactList
                  facts={{
                    review_score: assessment.review_score,
                    review_priority: assessment.review_priority,
                    policy_version: assessment.policy_version,
                  }}
                />
              )}
            </Section>
            <Triage view={view} />
            <Anomaly view={view} />
            <Audit view={view} onEvidence={openEvidence} />
          </div>
        )}
      </section>
      {evidenceId && (
        <EvidenceDrawer
          id={evidenceId}
          evidence={view.evidence.find((item) => item.id === evidenceId)}
          onClose={() => setEvidenceId(null)}
        />
      )}
    </>
  );
}
