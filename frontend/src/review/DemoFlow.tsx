/** A conceptual map; the console reads saved results rather than running this pipeline. */
export function DemoFlow() {
  return (
    <section className="demo-flow" aria-labelledby="demo-flow-title">
      <div className="demo-flow-caption">
        <h2 id="demo-flow-title">From evidence to a review</h2>
        <span>Previously generated results</span>
      </div>
      <ol aria-label="How evidence supports the review">
        <li>
          <strong>PDFs & photos</strong>
          <span>Claim evidence</span>
        </li>
        <li>
          <strong>Document AI + vision</strong>
          <span>Structured findings</span>
        </li>
        <li>
          <strong>LLM agent</strong>
          <span>Findings + policy evidence</span>
        </li>
        <li>
          <strong>Review guidance</strong>
          <span>Final decision stays human</span>
        </li>
      </ol>
    </section>
  );
}
