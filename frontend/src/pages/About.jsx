import TopBar from "../components/TopBar.jsx";

export default function About() {
  return (
    <>
      <TopBar title="About SkyCast" description="A historical airfare estimation model, not a live ticket booking or price-tracking system." />
      <section className="card">
        <h2>What is SkyCast?</h2>
        <p>
          SkyCast is a supervised regression system that estimates Indian domestic airfares from route, airline, cabin
          class, stops, duration, booking window, and time-of-day features.
        </p>
        <h2 style={{ marginTop: "1rem" }}>How the model works</h2>
        <p>
          A scikit-learn pipeline one-hot encodes categoricals with <code>handle_unknown=&quot;ignore&quot;</code>, scales
          numeric fields (including Haversine <code>distance_km</code> and coordinates), and predicts with the selected
          regressor (currently a tuned Random Forest on the saved run).
        </p>
        <h2 style={{ marginTop: "1rem" }}>Limitations</h2>
        <ul className="muted">
          <li>Training city names are six metros; other airports use geography + unknown-category handling.</li>
          <li>Seat type, baggage, lounge, meals, and aircraft type are not in the dataset and are not used.</li>
          <li>Estimates are not live fares and should not be treated as booking quotes.</li>
        </ul>
      </section>
    </>
  );
}
