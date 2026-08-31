import { NavLink } from "react-router-dom";

const LINKS = [
  ["/", "Dashboard"],
  ["/predict", "Predict Fare"],
  ["/insights", "Insights"],
  ["/model", "Model Performance"],
  ["/dataset", "Dataset"],
  ["/history", "Prediction History"],
  ["/about", "About"],
];

export default function Sidebar({ open, onClose }) {
  return (
    <aside className={`sidebar ${open ? "open" : ""}`}>
      <div className="brand">
        <div className="brand-mark" aria-hidden>
          ✈
        </div>
        <div>
          <h1>SKYCAST</h1>
          <p>AI Airfare Intelligence</p>
        </div>
      </div>
      <nav className="nav-list" aria-label="Primary">
        {LINKS.map(([to, label]) => (
          <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`} onClick={onClose}>
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-foot nav-list">
        <NavLink to="/settings" className="nav-link" onClick={onClose}>
          Settings
        </NavLink>
        <NavLink to="/help" className="nav-link" onClick={onClose}>
          Help
        </NavLink>
      </div>
    </aside>
  );
}
