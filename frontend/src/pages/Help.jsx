import TopBar from "../components/TopBar.jsx";

export default function Help() {
  return (
    <>
      <TopBar title="Help" description="Search a city, pick airline and cabin, then predict." />
      <section className="card">
        <ol>
          <li>Type in FROM / TO — for example Leh, Delhi, Bangalore — and select an airport result.</li>
          <li>Identical airports are blocked. Unknown spellings show “Location not found.”</li>
          <li>Start the API from the SkyCast folder before predicting.</li>
        </ol>
      </section>
    </>
  );
}
