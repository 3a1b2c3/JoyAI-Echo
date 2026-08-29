import { Navigate, Route, Routes } from "react-router";
import App from "@/App";
import { LlmParamsPage } from "@/pages/LlmParamsPage";
import { PromptStackerPage } from "@/pages/PromptStackerPage";
import { ROUTES } from "@/router/routes";
import { EventStackerPage } from "@/pages/EventStackerPage";
import { MemoryReviewPreviewPage } from "@/pages/MemoryReviewPreviewPage";
import { MemoryBankPreviewPage } from "@/pages/MemoryBankPreviewPage";

export function AppRouter() {
  return (
    <Routes>
      <Route path={ROUTES.home} element={<App />} />
      <Route path={ROUTES.llmParams} element={<LlmParamsPage />} />
      {import.meta.env.DEV ? (
        <>
          <Route path={ROUTES.promptStack} element={<PromptStackerPage />} />
          <Route path={ROUTES.eventStack} element={<EventStackerPage />} />
          <Route
            path={ROUTES.memoryReviewPreview}
            element={<MemoryReviewPreviewPage />}
          />
          <Route
            path={ROUTES.memoryBankPreview}
            element={<MemoryBankPreviewPage />}
          />
        </>
      ) : null}
      <Route path="*" element={<Navigate to={ROUTES.home} replace />} />
    </Routes>
  );
}
