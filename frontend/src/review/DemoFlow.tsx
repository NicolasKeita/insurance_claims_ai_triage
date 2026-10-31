import workflowIllustration from '../assets/review-workflow-en-v2.png';

/** The generated illustration explains the workflow; it is not claim evidence. */
export function DemoFlow() {
  return (
    <figure className="demo-flow" aria-label="Illustrated review workflow">
      <div className="demo-flow-stage">
        <img
          src={workflowIllustration}
          width={2172}
          height={724}
          alt="Claim file (PDFs and photos), Document AI and computer vision, LLM with policy evidence, then human review."
        />
        <a href={workflowIllustration} target="_blank" rel="noreferrer">
          <span className="visually-hidden">View full image</span>
          <span aria-hidden="true">↗</span>
        </a>
      </div>
    </figure>
  );
}
