export type FirstFrameQuestionIntent =
  | "confirm_uploaded"
  | "decline_upload"
  | "request_edit"
  | "confirm_edit_done"
  | "other";

const CONFIRM_UPLOADED = new Set([
  "需要上传,已上传完毕",
  "已上传完毕",
  "已上传",
  "确认上传",
]);

const DECLINE_UPLOAD = new Set(["不上传", "无需上传"]);

const REQUEST_EDIT = new Set([
  "我想修改/增删参考图",
  "我要修改/增删参考图",
  "需要上传",
]);

export function classifyFirstFrameQuestion(
  label: string,
  question?: string,
): FirstFrameQuestionIntent {
  const trimmed = label.trim();
  if (CONFIRM_UPLOADED.has(trimmed)) return "confirm_uploaded";
  if (DECLINE_UPLOAD.has(trimmed)) return "decline_upload";
  if (REQUEST_EDIT.has(trimmed)) return "request_edit";
  if (trimmed === "是" && (question ?? "").includes("参考图")) {
    return "confirm_edit_done";
  }
  return "other";
}
