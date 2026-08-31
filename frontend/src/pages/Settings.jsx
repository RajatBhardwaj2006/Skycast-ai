import TopBar from "../components/TopBar.jsx";

export default function Settings() {
  return (
    <>
      <TopBar title="Settings" description="API host is configured with VITE_API_URL (default http://127.0.0.1:8000)." />
      <section className="card">
        <p>No account or cloud sync. Predictions stay in session storage on this device.</p>
      </section>
    </>
  );
}
