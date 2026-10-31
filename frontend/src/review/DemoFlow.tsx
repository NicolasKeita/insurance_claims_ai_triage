import workflowIllustration from '../assets/review-workflow-v1.png';

/** The generated illustration explains the workflow; it is not claim evidence. */
export function DemoFlow() {
  return (
    <figure className="demo-flow" aria-label="Illustrated review workflow">
      <img
        src={workflowIllustration}
        width={2172}
        height={724}
        alt="PDFs and photos are analysed with Document AI and computer vision. Structured findings and policy evidence support LLM guidance for a human reviewer."
      />
      <figcaption>
        <span>Workflow illustration · saved results · final decision stays human</span>
        <a href={workflowIllustration} target="_blank" rel="noreferrer">
          View full image <span aria-hidden="true">↗</span>
        </a>
      </figcaption>
    </figure>
  );
}
