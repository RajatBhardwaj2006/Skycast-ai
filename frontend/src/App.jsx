import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import { useSessionHistory } from "./hooks/useSessionHistory";
import About from "./pages/About.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Dataset from "./pages/Dataset.jsx";
import Help from "./pages/Help.jsx";
import History from "./pages/History.jsx";
import Insights from "./pages/Insights.jsx";
import ModelPerformance from "./pages/ModelPerformance.jsx";
import Predict from "./pages/Predict.jsx";
import Settings from "./pages/Settings.jsx";

export default function App() {
  const history = useSessionHistory();
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard history={history.items} />} />
        <Route path="/predict" element={<Predict onPredicted={history.add} />} />
        <Route path="/insights" element={<Insights />} />
        <Route path="/model" element={<ModelPerformance />} />
        <Route path="/dataset" element={<Dataset />} />
        <Route path="/history" element={<History history={history.items} onClear={history.clear} />} />
        <Route path="/about" element={<About />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/help" element={<Help />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
