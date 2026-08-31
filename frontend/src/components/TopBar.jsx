export default function TopBar({ title, description }) {
  return (
    <header className="topbar">
      <div>
        <div className="eyebrow">SkyCast</div>
        <h1>{title}</h1>
        <p className="lede">{description}</p>
      </div>
    </header>
  );
}
