import { useState } from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar.jsx";

export default function Layout() {
  const [open, setOpen] = useState(false);
  return (
    <div className="app-shell">
      <button className="menu-btn" type="button" aria-label="Open menu" onClick={() => setOpen(true)}>
        ☰
      </button>
      <div className={`backdrop ${open ? "open" : ""}`} onClick={() => setOpen(false)} />
      <Sidebar open={open} onClose={() => setOpen(false)} />
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
