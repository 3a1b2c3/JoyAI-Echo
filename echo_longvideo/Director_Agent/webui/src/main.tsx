import React from "react";
import ReactDOM from "react-dom/client";

// import App from "./App";
// import { EventStackerPage } from "./pages/EventStackerPage";
// import { PromptStackerPage } from "./pages/PromptStackerPage";
import "./globals.css";
import "./i18n";
import { AppRouter } from "./router";
import { BrowserRouter } from "react-router";
import { ToastHost } from "./components/ui/Toast";

const root = document.getElementById("root");
if (!root) throw new Error("root element missing");

const routerBasename = import.meta.env.BASE_URL.replace(/\/$/, "");

function Root() {
  return (
    <BrowserRouter basename={routerBasename}>
      <AppRouter />
    </BrowserRouter>
  );
  // const path = window.location.pathname;
  // if (path === "/promptstack" || path.startsWith("/promptstack/")) {
  //   return <PromptStackerPage />;
  // }
  // if (path === "/eventstack" || path.startsWith("/eventstack/")) {
  //   return <EventStackerPage />;
  // }
  // return <App />;
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <Root />
    <ToastHost />
  </React.StrictMode>,
);
